# Design Note: Auto-Unpin Guardrail

| | |
|---|---|
| **Status** | Draft — for review |
| **Author** | Ryan Asar |
| **Created** | 2026-07-21 |
| **Scope** | `cf-quoter/src/quoter.ts` (QuoterDO alarm loop) |

## 1. Context & the incident

The quoter's most recurring failure mode is **cap-pinning**: symmetric full-size
touch-quoting accumulates one-sided inventory whenever there's directional flow,
position-skew can't unwind it if the thin book gives no passive fills, and the
market **pins at the ±maxPosition cap**. Pinned, it can only quote the reducing
side, so it stops earning the subsidy it's there for and holds naked directional
risk until a human notices (2026-07-20/21: MUSKNW and NHSALES both pinned at −40
with presence collapsed to 59% / 33%, flattened manually). See v2-learnings #1,
#3, #5e, #5f.

Manual remediation is the anti-pattern. This note encodes the fix as an
**invariant the system enforces itself.**

## 2. The invariant

> **INV-PIN:** No market remains pinned near the position cap with the reducing
> side failing to fill for more than `PIN_PATIENCE` cycles without the system
> taking a risk-reducing action; and a market that cannot be unpinned after
> `SHED_MAX_ATTEMPTS` remediation attempts is flattened and stopped.

## 3. State machine

Per cycle, after fills are detected and the kill-switch is checked:

```
        |pos| < PIN_FRAC·cap ───────────────► NORMAL   (reset counters, quote two-sided)
                 ▲                                │
   shed/skew     │                                │ |pos| ≥ PIN_FRAC·cap
   reduces below │                                ▼
   threshold     │                              PINNED  (pinnedCycles++)
                 │                                │
                 │           pinnedCycles ≥ PIN_PATIENCE
                 │                                ▼
                 └──────────────────────────── SHED     (cross the spread to
                    shedAttempts < MAX          reduce `size` contracts;
                                                 cancel resting quotes first)
                                                  │
                              shedAttempts ≥ MAX  ▼
                                              ESCALATE  (doStop → flatten + stop)
```

- **PINNED → SHED:** after `PIN_PATIENCE` stuck cycles (passive skew given its
  chance), place a **marketable** (`post_only=false`) order on the reducing side,
  priced through the touch, for `min(size, |pos|)` contracts. This is the
  deliberate "pay a small taker cost to shed risk" trade. Cancel our resting
  quotes first so we don't fight ourselves.
- **Unpin, don't flatten.** The shed only needs to drop `|pos|` below the pin
  threshold so *two-sided quoting resumes* — not to zero. Minimum intervention.
- **SHED → ESCALATE:** if repeated sheds can't reduce (illiquid book, #5e), the
  market is untradeable → `doStop` fully flattens and stops, freeing capital.

## 4. Config (tunable; poll ≈ 12s)

| const | value | meaning |
|---|---|---|
| `PIN_FRAC` | 0.9 | pinned when `|pos| ≥ 0.9·maxPosition` |
| `PIN_PATIENCE` | 10 | stuck cycles (~2 min) before the first shed |
| `SHED_MAX_ATTEMPTS` | 3 | crosses before escalating to flatten+stop (~6–8 min total) |

`pinnedCycles` resets to 0 whenever `|pos|` drops below the pin threshold, so a
*slowly-reducing* pin (passive skew working) never triggers — only a **stuck**
one does.

## 5. Safety properties

- **Only ever reduces risk.** The shed is always in the reducing direction, so it
  can move toward flat, never breach the cap.
- **No over-shedding into opposite inventory.** Detection keys off the
  *authoritative* position (`s.authPos`, refreshed each cycle by shadowReconcile),
  never the inferred `s.position` — which goes stale because `detectFills` can't
  track a taker shed. Each shed is additionally gated on a **fresh `position_fp`
  read** taken immediately before crossing, so a stale count can never cause a
  cross that flips the position past flat.
- **Bounded cost.** At most `SHED_MAX_ATTEMPTS` crosses of `size` contracts before
  escalation; the taker fee is paid only after passive reduction demonstrably
  failed for `PIN_PATIENCE` cycles.
- **Fail-safe.** If shadow reconciliation is down (`authPos` unavailable) or the
  fresh read fails, the guardrail does nothing and the deadline auto-flatten
  remains the backstop — it never acts on uncertain state.
- **Migration-safe.** New state fields default via `?? 0` for DOs already running.

## 6. Failure modes / limitations

- **One-sided book** (no touch on the reducing side to cross into): the shed
  can't price, so it no-ops that cycle; the deadline auto-flatten remains the
  backstop. `doStop`'s own flatten has the same constraint, so escalating
  wouldn't help — this is the honest floor.
- Crossing the spread realizes a small loss by design; that's the point (shed
  uncompensated risk you're no longer paid to hold).

## 7. Observability

- `/status` exposes `pinnedCycles` and `shedAttempts`.
- Structured logs: `unpin_shed` (side, n, px, attempt) and `unpin_escalate`.
- A future metric: count of shed/escalate events per market → how often the fleet
  self-heals vs how often selection should have avoided the market.

## 8. Rollout

Ship to the live fleet; it is conservative (acts only after ~2 min stuck, only
reduces risk). Watch `unpin_shed`/`unpin_escalate` logs on the next pin episode
to confirm it fires and unpins as designed.
