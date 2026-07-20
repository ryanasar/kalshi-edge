# Design Doc: Position Correctness

| | |
|---|---|
| **Status** | Draft — for review |
| **Author** | Ryan Asar |
| **Created** | 2026-07-20 |
| **Reviewers** | — |
| **Scope** | `cf-quoter/` (QuoterDO), Kalshi exchange client |

---

## 1. Context & problem

The live quoter (`cf-quoter/src/quoter.ts`) runs one Durable Object per market,
unattended, against a real-money exchange. Today it tracks its own `position`
and `cash` by **inference**: `detectFills()` diffs the exchange's resting-order
list against the DO's last-known legs, and any order that disappeared without
our cancel is *assumed* filled (`recordFill()`).

This derived-state approach is the root cause of every position-related incident
we've had:

- **Phantom fills.** An order can leave the resting list for reasons other than
  a fill — exchange-side cancel, expiry, self-trade-prevention. We count all of
  them as fills, corrupting `position`, which corrupts the kill-switch mark and
  the position-skew logic.
- **Double placement.** `restLimit()` sends a fresh random `client_order_id`
  per call. If a POST ack is lost but the exchange accepted the order, the next
  cycle retries and places a **second** order.
- **No running reconciliation.** `position_fp` (the exchange's authoritative
  count) is read only in `start()`. While running, tracked position can drift
  arbitrarily far from truth with nothing detecting it. This is how run-1 left
  orphaned positions.

For a system that moves money unattended, **position must be a reconciled fact,
not a computed guess.** This doc specifies the invariant, the reconciliation
protocol, the crash-recovery model, and the test/rollout plan to get there.

## 2. Goals / non-goals

**Goals**
- A stated, enforced consistency guarantee for position and fills (§4).
- Exactly-once fill accounting and exactly-once order placement.
- Deterministic crash recovery with no operator intervention.
- A pure, replayable decision core so the money path is testable without money.
- Observability that *proves* the invariant holds in production.

**Non-goals**
- Changing the trading strategy (skew, sizing, pool selection) — out of scope.
- New venues (Polymarket). Kalshi only.
- Sub-second latency. The poll-interval reconciliation model is sufficient;
  event-driven quoting is a separate, later doc.
- An external event store (Kafka/DB-as-WAL) — see Alternatives (§10); DO storage
  is already durable and strongly consistent per-DO.

## 3. Background: why the current design is insufficient

`alarm()` today: `detectFills → bestQuote → requote(bid) → requote(ask)`, then
persist a single JSON state blob. The blob is the *only* record; there is no
append-only history, no exchange reconciliation, no idempotency, and the
decision logic is interleaved with I/O (so it can't be replayed or unit-tested).
The platform (Cloudflare DO single-threaded execution) saves us from concurrent
re-entry, but not from *semantic* drift between our state and the exchange's.

## 4. The invariant (the guarantee we make)

The system SHALL maintain, per market:

- **INV-1 (Convergence).** At the end of every reconciliation cycle, the DO's
  folded net position equals the exchange's authoritative `position_fp`, **or**
  the market is in SAFE MODE.
- **INV-2 (Fill exactly-once).** Every exchange fill is applied to the local
  ledger exactly once, regardless of retries, restarts, or duplicate delivery.
- **INV-3 (Placement exactly-once).** Each order *intent* results in at most one
  live exchange order.
- **INV-4 (Crash safety).** A crash at any point in the order lifecycle leaves
  the system able to re-establish INV-1–3 on restart without operator action,
  within RTO (§7).
- **INV-5 (Bounded risk under divergence).** While folded position ≠ authoritative,
  the system SHALL NOT place risk-increasing orders (SAFE MODE).

INV-5 is the safety backstop that makes the others tolerable: correctness may
lag reality by up to one cycle, but during that lag we never *add* exposure.

## 5. Design

### 5.1 Source of truth

The **exchange is the linearizable source of truth** for position and fills.
Local state is a durable *cache* reconciled against it. Relevant endpoints:

- `GET /portfolio/fills` — authoritative fill history, each with an
  exchange-assigned `trade_id`/`fill_id` and timestamp. Paginated; supports an
  incremental cursor (min_ts / cursor).
- `GET /portfolio/positions` — `position_fp` per market (secondary check).
- `GET /portfolio/orders?status=resting` — live resting orders (for placement
  reconciliation).

### 5.2 Event-sourced fill ledger (WAL)

Replace the single state blob with an **append-only ledger** in DO storage:

```
LedgerEntry =
  | { kind: "fill",  fill_id, ticker, side, count, price, exch_ts }
  | { kind: "intent", intent_seq, ticker, side, price, size, client_order_id }
```

- `position = fold(ledger)`, `cash = fold(ledger)` — deterministic, auditable.
- **Fills are keyed by exchange `fill_id`**; applying the same `fill_id` twice is
  a no-op. This is what buys INV-2 across retries/restarts/duplicate delivery.
- On startup: replay the ledger to reconstruct position, then reconcile (§5.4).
- **Snapshotting/compaction:** periodically write a `snapshot(position, cash,
  cursor)` and truncate ledger entries older than it, bounding storage (standard
  event-sourcing compaction; DO storage is finite).

### 5.3 Deterministic idempotency keys (INV-3)

```
client_order_id = hash(ticker, side, price_fp, size, intent_seq)
```

`intent_seq` is a monotonic counter persisted in DO state, bumped once per
*distinct* intent. A retry of the *same* intent reuses the same
`client_order_id`, so the exchange dedups it rather than opening a second order.
Distinct intents (a genuinely new quote) get a new `intent_seq` → new key.

> **Must-verify (§9):** Kalshi's `client_order_id` idempotency semantics — does a
> duplicate key return the existing order, or reject? INV-3's mechanism depends
> on this. If Kalshi does *not* honor idempotency keys, fall back to
> reconcile-before-place (query resting orders + fills for this intent's key
> before issuing).

### 5.4 Reconciliation protocol

Every cycle **and** on startup:

1. Pull new fills via `/portfolio/fills` since `last_fill_cursor`.
2. Apply each new fill to the ledger (dedup by `fill_id`); advance cursor.
3. Fold ledger → `tracked_position`. Fetch `position_fp` → `authoritative`.
4. If `tracked_position == authoritative` → **INV-1 holds**, exit SAFE MODE if
   set, proceed to quote.
5. If not:
   - Enter **SAFE MODE** (INV-5): cancel resting orders, place no risk-increasing
     orders.
   - The usual cause is a fill the fills-feed hasn't surfaced yet; re-pull fills.
   - If reconciled within `K` cycles → resume.
   - If still divergent after `K` cycles → **flatten to the authoritative
     position** (trust the exchange), emit a drift alert, page.

Reconciliation compares **ledger-folded position** (primary; it has fill-level
detail) against **`position_fp`** (secondary cross-check), since `position_fp`
may lag or lead the fills feed by a tick.

### 5.5 Pure decision core (replay + testability)

Refactor the heart of `alarm()` into a **pure function**:

```
decide(state, event) -> { state', actions[] }

event  ∈ { BookUpdate(bid,ask,sizes) | FillsObserved(fills[])
         | PositionObserved(position_fp) | Tick(now_ms) }
action ∈ { Place(intent) | Cancel(order_id) | Flatten | EnterSafeMode | Alert }
```

`decide()` performs **no I/O** and reads the clock **only** from `Tick.now_ms`
(no `Date.now()` inside). `alarm()` becomes a thin shell:

```
events = gather()      // I/O: fills, position, book
(state, actions) = decide(state, events)   // pure
execute(actions)       // I/O: place/cancel
persist(state, ledger) // durable
```

Because `decide()` is pure and clock-injected, a recorded event log **replays
bit-for-bit** → deterministic debugging, regression tests, and crash injection
(§8) without touching the exchange or spending money.

## 6. Consistency model

- **Truth:** exchange position is authoritative and linearizable.
- **Local:** an eventually-consistent durable cache with a **convergence bound of
  one reconciliation interval** (`pollSeconds`): within one cycle of any fill,
  `tracked == authoritative`, or the market is in SAFE MODE.
- **RPO = 0 for fills.** No fill is ever lost: it is durable on the exchange and
  re-derived via the fills cursor even if the DO's ledger is behind or was
  destroyed. The ledger is a cache, not the system of record.
- **RTO = one reconciliation cycle** after restart: replay ledger → pull fills
  since cursor → compare → resume or SAFE MODE.

## 7. Failure analysis (crash at each lifecycle stage)

| # | Crash point | Recovery on restart | Invariant |
|---|---|---|---|
| 1 | After `decide` chose Place, before POST | No order exists, no intent persisted; `decide` re-issues with same deterministic key | INV-3 |
| 2 | After POST sent, before ack (ack lost) | Order *may* exist; reconciliation finds it by `client_order_id` in resting/fills; no double-place | INV-3 |
| 3 | After ack, before persisting intent | Ledger missing the intent, but reconciliation vs resting-orders + fills rebuilds truth | INV-1 |
| 4 | After a fill, before observing it | Fill is durable on exchange; next cursor pull applies it once (dedup by `fill_id`) | INV-2 |
| 5 | During flatten | Position ≠ flat on restart; reconcile → re-issue flatten (idempotent) | INV-1/5 |
| 6 | Alarm stops firing entirely | Heartbeat + supervisor (separate doc) restarts the DO; reconciliation on restart | INV-4 |

The load-bearing insight: because truth lives on the exchange and the ledger
dedups by **exchange-assigned ids**, no crash can lose or double-count a fill.
The worst case is a bounded reconciliation delay, and SAFE MODE (INV-5) prevents
new risk during it.

## 8. Test plan

The money path is currently untested. This design makes it testable; the test
suite is a **deliverable, not an afterthought.**

- **Unit:** ledger fold; idempotent fill application (apply `fill_id` twice → no
  change); `client_order_id` determinism; SAFE MODE transitions.
- **Property-based (fast-check / hypothesis-style):** for any random sequence of
  fills/cancels/crashes, `fold(ledger) == replay(events)`; every `fill_id`
  applied exactly once (INV-2); position never diverges without SAFE MODE.
- **Deterministic replay + crash injection (the R1 harness):** record a real
  session's event log; replay it through `decide()`; parametrically inject a
  crash at each lifecycle stage (§7) and assert INV-1–5 hold on recovery.
- **Chaos:** inject lost acks, duplicate fills, out-of-order fills, exchange
  429/5xx, and `position_fp` disagreeing with the fills feed.
- **Integration:** against a recorded-fixture mock of the Kalshi endpoints (and
  the demo environment if fills are available there).

## 9. Observability

- **Metrics** (→ Cloudflare Analytics Engine or Postgres): `position_drift`
  (tracked − authoritative) per market, `reconciliation_latency`, `fills_applied`,
  `safe_mode_entries`, `dup_placement_prevented`, `flatten_failures`.
- **Structured logs** with a per-cycle trace id spanning gather→decide→execute.
- **Alerts:** drift persists > `K` cycles; SAFE MODE entered; flatten failed;
  reconciliation error-rate spike.

A drift metric pinned at 0 in production is the *evidence* that INV-1 holds — the
thing the current system cannot produce.

## 10. Alternatives considered

- **Keep inference, add a periodic `position_fp` check.** Rejected: closes the
  detection gap but gives no exactly-once fills, no crash-safe recovery, and no
  audit/replay log.
- **External event store (Kafka / Postgres WAL).** Rejected: DO storage is
  already durable and strongly consistent *per DO*, which is exactly the scope of
  one market's ledger. An external store adds operational surface and a network
  dependency without solving a problem we have at this scale.
- **Exchange as sole truth, no local ledger.** Rejected: we need a durable local
  log to (a) replay/test the decision core, (b) decide without a synchronous
  round-trip per action, and (c) audit.

## 11. Open questions / must-verify before implementation

**Findings from the read-only spike (2026-07-20) — most items resolved:**

- ✅ **`/portfolio/fills` schema.** Each fill carries a unique **`fill_id`** (plus
  `order_id`, `count_fp`, `side`, `yes/no_price_dollars`, `fee_cost`, `ts`) — this
  is the INV-2 dedup key. The endpoint returns a **`cursor`** and accepts
  **`min_ts`** and **`ticker`** filters, so incremental per-market reconciliation
  (§5.4) is directly supported. Feed confirmed live/fresh.
- ✅ **`client_order_id` is stored + queryable** on `/portfolio/orders`. Fills lack
  it directly but carry `order_id`, so fill→order→`client_order_id` is one hop.
  This makes the §5.3 **reconcile-before-place fallback viable regardless** of
  native idempotency enforcement.
- ✅ **`position_fp` carries `last_updated_ts`** — §5.4 can reason about staleness.
- ✅ **DO storage** is ample for a per-market ledger with periodic snapshot/compaction
  (§5.2); not a constraint at fleet size.

**Still open:**

1. **Native `client_order_id` idempotency** (return-existing vs reject vs
   create-duplicate) — needs a *tiny live test order* to settle; **non-blocking**
   because the reconcile-before-place fallback (§5.3) already satisfies INV-3.
2. **Fills freshness vs `position_fp` timing** — do they ever disagree, and by how
   long? Answered by the shadow-mode drift metric (§12 step 2), not a spike.

## 12. Rollout plan

1. **Verify exchange semantics** (§11) — a read-only spike. Gate: INV-3 mechanism
   confirmed or fallback chosen.
2. **Shadow mode.** Ship the ledger + reconciliation running *alongside* the
   existing logic on the live fleet: compute drift, emit metrics, but do not yet
   let it drive orders. Gate: quantify how often the *old* inference was wrong,
   and confirm reconciled drift ≈ 0.
3. **Enforce on canary.** Enable reconciled-ledger + SAFE MODE on **one** market.
   Gate: no missed fills, no double-placements, drift alert clean for N days.
4. **Fleet rollout**, then **snapshotting/compaction**.

## 13. Milestones

1. Exchange-semantics spike (§11).
2. `decide()` refactor + event types (§5.5).
3. Ledger + reconciliation in **shadow mode** + metrics (§5.2, §5.4, §9).
4. Replay + crash-injection test harness (§8).
5. Enforcement (SAFE MODE) + canary (§12).
6. Snapshot/compaction.
