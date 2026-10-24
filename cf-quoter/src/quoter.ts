/**
 * src/quoter.ts — QuoterDO: an always-on two-sided quoter for ONE market.
 *
 * Port of the proven Python quoter.py, restructured for a Durable Object:
 * instead of a sleep loop, each quoting cycle runs in alarm(), which reschedules
 * itself every pollSeconds. All state is persisted to DO storage, so an eviction
 * or crash resumes cleanly (the exchange remembers our resting orders; we reload
 * our tracking of them).
 *
 * REAL MONEY, UNATTENDED. Safety (mirrors quoter.py, and matters MORE here
 * because nothing else is watching):
 *   • size/maxPosition caps bound exposure to a few dollars.
 *   • deadline (endTime): stops + flattens after the configured window.
 *   • max-loss kill: marked PnL ≤ −maxLoss → flatten + stop.
 *   • stop()/deadline/kill ALL flatten: cancel resting orders, then close the
 *     exchange's AUTHORITATIVE position (position_fp) with marketable orders,
 *     re-checking each cycle until it reads flat (bounded attempts).
 *   • start() refuses to begin unless the account is flat (no resting orders,
 *     no position) — never resume on top of unknown risk.
 *   • self-tracks position from its own fills, persisting last-seen remaining so
 *     an alarm retry can't double-count.
 */

import { DurableObject } from "cloudflare:workers";
import { Kalshi, KalshiHttpError, fp, type Quote } from "./kalshi";
import { TICK, newFillsOnGoneLeg, skewPrices } from "./logic";
import type { Env } from "./index";

const MAKER_RATE = 0.0175;
const TAKER_RATE = 0.07;
// v2 position-skew: once |inventory| reaches this fraction of maxPosition, start
// improving the REDUCING side by one tick so the position works back to flat
// instead of pinning at the cap (§ v2 learnings — the USPPIYOY pin).
const SKEW_SOFT_FRAC = 0.6;
// Auto-unpin guardrail (docs/design/auto-unpin.md, INV-PIN): when inventory
// sticks at the cap and the reducing side won't fill passively, cross the spread
// to shed risk; after repeated failures, flatten + stop. Turns the manual
// "notice the pin and flatten" incident into an enforced invariant.
const PIN_FRAC = 0.9;          // pinned when |pos| >= 0.9 * maxPosition
const PIN_PATIENCE = 10;       // stuck cycles (~2 min @ 12s poll) before the first shed
const SHED_MAX_ATTEMPTS = 3;   // crosses before escalating to flatten + stop
// Verified flatten: marketable close attempts (one per cycle, each re-reading
// position_fp) before giving up and leaving the position open, loudly.
const FLATTEN_MAX_ATTEMPTS = 5;

type Side = "bid" | "ask";
type Slot = "bid" | "ask" | "shed";

// `id` is the exchange order_id; "" means the POST was sent but never acked
// (PENDING). `coid` is our client_order_id — the idempotency key, generated and
// persisted BEFORE the POST so a pending leg can be resolved by lookup. `side`
// is the order's direction (the shed slot can be either). `qty` is the size we
// placed, so a leg that leaves the resting list can be reconciled against the
// exchange's own fill count. side/qty are optional only for legs persisted
// before they existed.
interface Leg { id: string; coid?: string; side?: Side; price: string; rem: number; qty?: number; }
interface State {
  ticker: string;
  size: number;
  maxPosition: number;
  maxLoss: number;       // dollars
  pollSeconds: number;
  endTime: number;       // epoch ms
  running: boolean;
  stopReason: string;
  position: number;
  cash: number;
  fees: number;
  fills: number;
  cycles: number;
  atBest: number;
  bid: Leg | null;
  ask: Leg | null;
  // The auto-unpin taker order, tracked like a quote leg so its fills move the
  // inferred position, and cancelled after one cycle so a partially-filled
  // remainder can't sit on the book untracked.
  shed: Leg | null;
  // SHADOW MODE (observational; NEVER drives orders). Position folded from the
  // exchange fills feed, plus the authoritative position_fp, so we can measure
  // drift between our INFERRED position and exchange truth on the live fleet
  // before trusting a reconciled ledger to drive decisions. Step 2 of
  // docs/design/position-correctness.md.
  shadowPos: number;    // net YES position folded from the fills feed
  shadowCash: number;   // cash folded from the fills feed (best-effort)
  authPos: number;      // exchange authoritative position_fp
  shadowTs: number;     // epoch ms of last successful reconcile (0 = never)
  shadowTruncated: boolean; // fills feed exceeded the page cap → shadowPos not updated
  // Auto-unpin guardrail state (docs/design/auto-unpin.md).
  pinnedCycles: number; // consecutive cycles stuck at/above the pin threshold
  shedAttempts: number; // cross-the-spread shed attempts this pin episode
  // Verified-flatten state: after a stop, the alarm keeps running in this mode
  // until position_fp reads 0 or attempts run out.
  flattening: boolean;
  flattenAttempts: number;
}

export interface StartConfig {
  ticker: string;
  size?: number;
  maxPosition?: number;
  maxLoss?: number;
  pollSeconds?: number;
  minutes?: number;
}

function remaining(o: any): number {
  const r = o.remaining_count ?? o.remaining_count_fp;
  if (r != null) return Number(r);
  return Number(o.initial_count_fp ?? o.initial_count ?? 0)
    - Number(o.fill_count_fp ?? o.fill_count ?? 0);
}

export class QuoterDO extends DurableObject<Env> {
  private k?: Kalshi;

  private async kalshi(): Promise<Kalshi> {
    if (!this.k) {
      this.k = await Kalshi.create(this.env.KALSHI_API_KEY_ID, this.env.KALSHI_PRIVATE_KEY_B64);
    }
    return this.k;
  }

  // Resting orders for THIS DO's market only. The exchange returns every order
  // on the account, but with one DO per ticker we must never touch another
  // market's orders — the whole point of per-ticker scoping. Every place that
  // cancels or flat-checks goes through this filter (the server-side ticker
  // param scopes the query; the client-side filter is belt-and-suspenders).
  private async myOrders(k: Kalshi, ticker: string): Promise<any[]> {
    return (await k.restingOrders(ticker)).filter(
      (o) => (o.ticker ?? o.market_ticker) === ticker,
    );
  }

  // --- control (RPC) ------------------------------------------------------
  async start(cfg: StartConfig): Promise<object> {
    const k = await this.kalshi();
    // Flat-check is PER TICKER now: another market's resting orders are none of
    // this DO's business (the old account-wide check made simultaneous markets
    // impossible — the second start() always saw the first market's orders).
    if ((await this.myOrders(k, cfg.ticker)).length) throw new Error(`resting orders already in ${cfg.ticker}`);
    // Kalshi returns the signed contract count as `position_fp` (a string) — NOT
    // `position` (which doesn't exist, so reading it silently sees 0 and would let
    // us start ON TOP of an open position). This is the field bug that left run-1
    // econ positions orphaned; the flat-check must read position_fp.
    const pos = await k.positions(cfg.ticker);
    if (pos.some((p) => Number(p.position_fp ?? p.position ?? 0) !== 0)) throw new Error(`open position in ${cfg.ticker}`);

    // Buying-power guard: the $147 account is SHARED across every market, so a
    // new market must not commit more than the account can currently cover.
    // Worst case this market holds a full one-sided position of `maxPosition`
    // contracts (~$1 each), so require at least that much free balance now.
    const need = cfg.maxPosition ?? 4;
    const bal = Number(await k.balanceDollars());
    if (bal < need) throw new Error(`balance $${bal.toFixed(2)} < ~$${need} needed for maxPosition ${need}`);

    const minutes = cfg.minutes ?? 180;
    const s: State = {
      ticker: cfg.ticker,
      size: cfg.size ?? 2,
      maxPosition: cfg.maxPosition ?? 4,
      maxLoss: cfg.maxLoss ?? 5,
      pollSeconds: cfg.pollSeconds ?? 10,
      endTime: Date.now() + minutes * 60_000,
      running: true, stopReason: "",
      position: 0, cash: 0, fees: 0, fills: 0, cycles: 0, atBest: 0,
      shadowPos: 0, shadowCash: 0, authPos: 0, shadowTs: 0, shadowTruncated: false,
      pinnedCycles: 0, shedAttempts: 0,
      flattening: false, flattenAttempts: 0,
      bid: null, ask: null, shed: null,
    };
    await this.ctx.storage.put("state", s);
    await this.ctx.storage.setAlarm(Date.now() + 1000); // first cycle shortly
    return { started: true, ...this.snapshot(s) };
  }

  async stop(): Promise<object> {
    const s = await this.ctx.storage.get<State>("state");
    if (!s) return { stopped: true, note: "no active session" };
    await this.doStop(s, "manual");
    return { stopped: true, ...this.snapshot(s) };
  }

  async status(): Promise<object> {
    const s = await this.ctx.storage.get<State>("state");
    return s ? this.snapshot(s) : { running: false, note: "never started" };
  }

  private snapshot(s: State) {
    return {
      ticker: s.ticker, running: s.running, stopReason: s.stopReason,
      flattening: s.flattening ?? false, flattenAttempts: s.flattenAttempts ?? 0,
      position: s.position, fills: s.fills, cycles: s.cycles, atBest: s.atBest,
      atBestPct: s.cycles ? Math.round((100 * s.atBest) / s.cycles) : 0,
      cash: +s.cash.toFixed(4), fees: +s.fees.toFixed(4),
      // shadow-mode drift (observational). driftAuth ≈ 0 validates the fold;
      // driftInferred is the actual bug metric: how far the OLD inferred
      // position is from exchange truth.
      // ?? 0 guards the schema migration: DOs already running when this deploys
      // have persisted State without these fields; shadowReconcile backfills them
      // on the first post-deploy alarm.
      shadowPos: +(s.shadowPos ?? 0).toFixed(2), authPos: +(s.authPos ?? 0).toFixed(2),
      driftAuth: +((s.shadowPos ?? 0) - (s.authPos ?? 0)).toFixed(2),
      driftInferred: +(s.position - (s.authPos ?? 0)).toFixed(2),
      shadowAgeSec: s.shadowTs ? Math.round((Date.now() - s.shadowTs) / 1000) : null,
      shadowTruncated: s.shadowTruncated ?? false,
      pinnedCycles: s.pinnedCycles ?? 0, shedAttempts: s.shedAttempts ?? 0,
      endsInMin: Math.max(0, Math.round((s.endTime - Date.now()) / 60_000)),
    };
  }

  // --- the loop -----------------------------------------------------------
  async alarm(): Promise<void> {
    const s = await this.ctx.storage.get<State>("state");
    if (!s) return;
    const k = await this.kalshi();
    if (s.flattening) {
      try { await this.flattenStep(k, s); }
      catch (e) {
        console.log(`flatten step error (${s.ticker}): ${(e as Error).message}`);
        await this.ctx.storage.put("state", s);
        await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
      }
      return;
    }
    if (!s.running) return;
    try {
      if (Date.now() >= s.endTime) { await this.doStop(s, "deadline"); return; }
      await this.detectFills(k, s);
      await this.shadowReconcile(k, s);   // observational; self-isolated, never throws into the loop
      const q = await k.bestQuote(s.ticker);
      if (q.bid && q.ask) {
        const mid = (Number(q.bid) + Number(q.ask)) / 2;
        const pnl = s.cash + s.position * mid - s.fees;
        s.cycles++;
        if (s.bid && s.ask && fp(s.bid.price) === fp(q.bid) && fp(s.ask.price) === fp(q.ask)) s.atBest++;
        if (pnl <= -s.maxLoss) { await this.doStop(s, `killswitch pnl=${pnl.toFixed(2)}`); return; }
        // Auto-unpin: if stuck at the cap with the reducing side not filling,
        // cross the spread to shed risk (or, after repeated failures, flatten +
        // stop). If it acts, skip normal requoting this cycle to let it settle.
        if (await this.maybeUnpin(k, s, q)) {
          if (!s.running) return;                    // escalated to flatten + stop
          await this.ctx.storage.put("state", s);
          await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
          return;
        }
        // Position-skew the PRICES (logic.ts): below the soft threshold both
        // sides join best; past it, the reducing side sits one tick better than
        // the best COMPETING price — and holds there once it's alone at best,
        // instead of re-improving on its own quote every cycle. Maker-only, never
        // crosses (post_only would reject a cross anyway).
        const leg = (l: Leg | null) => (l && l.id ? { price: Number(l.price), rem: l.rem } : null);
        const { bidPx, askPx } = skewPrices({
          position: s.position, maxPosition: s.maxPosition, softFrac: SKEW_SOFT_FRAC,
          bid: Number(q.bid), ask: Number(q.ask), bidSize: q.bidSize, askSize: q.askSize,
          ourBid: leg(s.bid), ourAsk: leg(s.ask),
        });
        // Pass the ROOM on each side (contracts until the ±cap), so requote can
        // size the order down and a full fill can never breach maxPosition.
        await this.requote(k, s, "bid", s.maxPosition - s.position, fp(bidPx));
        await this.requote(k, s, "ask", s.maxPosition + s.position, fp(askPx));
      }
      await this.ctx.storage.put("state", s);
      await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
    } catch (e) {
      // transient (network/rate-limit): persist + reschedule so the loop
      // survives a blip. Placing no order on error is safe; fills reconcile
      // next cycle from the exchange's own resting-order list.
      console.log("alarm error:", (e as Error).message);
      await this.ctx.storage.put("state", s);
      await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
    }
  }

  private async detectFills(k: Kalshi, s: State): Promise<void> {
    const live = new Map<string, any>();
    for (const o of await this.myOrders(k, s.ticker)) live.set(o.order_id ?? o.id, o);
    // Every recent order for this market in ANY status, fetched lazily — only
    // when a leg is pending or has left the resting list.
    let all: any[] | null = null;
    const history = async () => (all ??= await k.orders(s.ticker));

    for (const slot of ["bid", "ask", "shed"] as const) {
      const o = s[slot];
      if (!o) continue;
      const side: Side = o.side ?? (slot === "shed" ? "ask" : slot);
      const rate = slot === "shed" ? TAKER_RATE : MAKER_RATE;
      // PENDING leg (POST sent, ack lost): resolve it by idempotency key against
      // orders in ANY status — not just resting. A post_only order can land and
      // fully fill inside one poll interval; checking only the resting list
      // would conclude "never landed", drop its fills, and quote again on top.
      if (!o.id) {
        const hit = o.coid ? (await history()).find((x) => x.client_order_id === o.coid) : undefined;
        console.log(JSON.stringify({ evt: "pending_resolve", ticker: s.ticker, slot,
          coid: o.coid, found: !!hit, status: hit?.status }));
        if (!hit) { s[slot] = null; continue; }
        o.id = hit.order_id ?? hit.id;
      }
      if (live.has(o.id)) {
        const rem = remaining(live.get(o.id));
        if (rem < o.rem) { this.recordFill(s, side, o.price, o.rem - rem, rate); o.rem = rem; }
        if (rem <= 0) s[slot] = null;
      } else {
        // Gone, and we didn't cancel it. Count what the exchange says actually
        // filled — an exchange-side cancel / expiry / self-trade-prevention
        // removal is NOT a fill. Falls back to "gone ⇒ filled" only when the
        // order's record can't be found.
        const rec = (await history()).find((x) => (x.order_id ?? x.id) === o.id);
        this.recordFill(s, side, o.price, newFillsOnGoneLeg(o, rec), rate);
        s[slot] = null;
      }
    }

    // The shed is a one-cycle taker order: it got its fills above; cancel any
    // remainder so it can't rest on the book untracked at its aggressive price.
    if (s.shed) {
      await k.cancel(s.shed.id).catch((e) =>
        console.log(`shed remainder cancel (${s.ticker}): ${(e as Error).message}`));
      s.shed = null;
    }
  }

  // --- SHADOW MODE: reconcile against exchange truth, measure drift ----------
  // Observational ONLY — never places, cancels, or changes any order. Folds this
  // market's fills feed into a signed YES position and compares (a) that ledger
  // position and (b) our INFERRED position against the exchange's authoritative
  // position_fp. Re-folded from scratch each cycle, so it is idempotent by
  // construction (no fill can be double-counted). Fully wrapped: any error is
  // swallowed so shadow reconciliation can NEVER disturb the trading path.
  // This is the safe first step (shadow before enforce) of
  // docs/design/position-correctness.md — it produces the drift evidence the
  // current inferred-position design cannot.
  private async shadowReconcile(k: Kalshi, s: State): Promise<void> {
    // Authoritative position FIRST, in its own guard: maybeUnpin keys off
    // authPos, so it must not go stale just because the fills feed failed.
    try {
      s.authPos = await k.positionFp(s.ticker);
      s.shadowTs = Date.now();
    } catch (e) {
      console.log(`position_fp read error (${s.ticker}): ${(e as Error).message}`);
    }
    try {
      const { fills, truncated } = await k.fills(s.ticker);
      s.shadowTruncated = truncated;
      if (truncated) {
        // A partial fold is a wrong position, not an approximation. Keep the
        // last good shadowPos and say so rather than publish a bad number.
        console.log(JSON.stringify({ evt: "shadow_truncated", ticker: s.ticker, n_fills: fills.length }));
        return;
      }
      let pos = 0, cash = 0;
      for (const f of fills) {
        const n = Number(f.count_fp ?? 0);
        const isBuy = f.action === "buy";
        // Signed YES position, VALIDATED against position_fp on 4 live markets
        // (buildperms −80, usretail −10, musknw −18.86, crunchwrap +10). The
        // quoter only ever posts YES bids and YES asks, so every fill is
        // action=buy (a YES bid → +YES) or action=sell (a YES ask, which Kalshi
        // records as sell/no → −YES). Sign is a SINGLE factor — do NOT also
        // multiply by outcome_side (it's collinear here, so that double-counts
        // the sign; that was the bug caught in pre-deploy validation). Revisit
        // if NO-side orders are ever added.
        pos += (isBuy ? 1 : -1) * n;
        cash += (isBuy ? -1 : 1) * n * Number(f.yes_price_dollars) - Number(f.fee_cost ?? 0);
      }
      s.shadowPos = pos;
      s.shadowCash = cash;
      // Structured line for `wrangler tail` → drift history / metrics extraction.
      console.log(JSON.stringify({
        evt: "shadow_reconcile", ticker: s.ticker,
        inferred: +s.position.toFixed(2), ledger: +pos.toFixed(2), auth: +s.authPos.toFixed(2),
        drift_auth: +(pos - s.authPos).toFixed(2),          // ledger vs exchange — should be ~0 (validates fold)
        drift_inferred: +(s.position - s.authPos).toFixed(2), // OLD inference vs exchange — the bug we're measuring
        n_fills: fills.length,
      }));
    } catch (e) {
      console.log(`shadow_reconcile error (${s.ticker}): ${(e as Error).message}`);
    }
  }

  private recordFill(s: State, side: Side, price: string, n: number, rate = MAKER_RATE): void {
    if (n <= 0) return;
    const p = Number(price);
    if (side === "bid") { s.position += n; s.cash -= p * n; }
    else { s.position -= n; s.cash += p * n; }
    s.fees += rate * p * (1 - p) * n;
    s.fills++;
    console.log(`FILL ${side} ${n}@${price} -> pos=${s.position} cash=${s.cash.toFixed(4)}`);
  }

  // Idempotent placement into a slot: write-ahead the intent (with its
  // client_order_id) to durable storage BEFORE the POST. If the ack is lost — or
  // the DO is evicted mid-request — the next cycle finds a PENDING leg and
  // resolves it by coid in detectFills instead of placing a duplicate order.
  private async place(k: Kalshi, s: State, slot: Slot, side: Side, price: string,
                      n: number, postOnly: boolean): Promise<void> {
    const coid = crypto.randomUUID();
    s[slot] = { id: "", coid, side, price: fp(price), rem: n, qty: n };
    await this.ctx.storage.put("state", s);
    try {
      s[slot]!.id = await k.restLimit(s.ticker, side, fp(price), n, postOnly, coid);
    } catch (e) {
      if (e instanceof KalshiHttpError && e.status >= 400 && e.status < 500) {
        s[slot] = null;   // definite reject (e.g. post_only crossed a racing quote): nothing was placed
        console.log(`skip ${slot} ${side}@${price}: ${e.message}`);
      } else {
        // Ambiguous (network/timeout/5xx): the order MAY exist. Keep the leg
        // pending; the next cycle resolves it by coid rather than re-placing.
        console.log(`pending ${slot} ${side}@${price} coid=${coid}: ${(e as Error).message}`);
      }
    }
  }

  // `room` = contracts allowed on this side before hitting the ±maxPosition cap.
  // We quote min(size, room), so even a full fill can't breach the cap — this is
  // the fix for the 53-vs-40 overshoot (the old code checked the cap but still
  // placed a full-size order, so effective cap was maxPosition + size).
  private async requote(k: Kalshi, s: State, tag: Side, room: number, price: string): Promise<void> {
    const want = Math.min(s.size, Math.max(0, room));
    const o = s[tag];
    if (want <= 0) { if (o) { await k.cancel(o.id); s[tag] = null; } return; }
    if (o && fp(o.price) === fp(price) && o.rem <= room) return; // at best & can't breach — keep it
    if (o) { await k.cancel(o.id); s[tag] = null; }
    await this.place(k, s, tag, tag, price, want, true);
  }

  // --- AUTO-UNPIN GUARDRAIL (docs/design/auto-unpin.md, INV-PIN) -----------
  // Returns true iff it took a remediation action this cycle (a shed or an
  // escalate), in which case the caller skips normal requoting. Only ever
  // REDUCES risk — the shed is always in the reducing direction, so it can move
  // toward flat but never breach the cap.
  private async maybeUnpin(k: Kalshi, s: State, q: Quote): Promise<boolean> {
    const cap = s.maxPosition;
    // DETECT on the AUTHORITATIVE position (shadowReconcile refreshes s.authPos
    // every cycle), not the inferred s.position, which can lag exchange truth.
    if (Math.abs(s.authPos ?? 0) < cap * PIN_FRAC) {   // not pinned → healthy, reset
      s.pinnedCycles = 0;
      s.shedAttempts = 0;
      return false;
    }
    s.pinnedCycles = (s.pinnedCycles ?? 0) + 1;
    if (s.pinnedCycles < PIN_PATIENCE) return false;   // give passive skew its chance first

    // Repeated crosses couldn't unpin us → the market is untradeable (thin book,
    // no counterparty on the reducing side). Flatten + stop, freeing the capital.
    if ((s.shedAttempts ?? 0) >= SHED_MAX_ATTEMPTS) {
      console.log(JSON.stringify({ evt: "unpin_escalate", ticker: s.ticker,
        authPos: s.authPos, shedAttempts: s.shedAttempts }));
      await this.doStop(s, "pin-unrecoverable");
      return true;
    }

    // Confirm the pin on a FRESH authoritative read before crossing, so a stale
    // count can never make us over-shed past flat into opposite inventory. Also
    // re-syncs the inferred position to truth while we hold it.
    let pos: number;
    try { pos = await k.positionFp(s.ticker); }
    catch { return false; }                            // can't confirm truth → don't cross this cycle
    s.position = pos;
    if (Math.abs(pos) < cap * PIN_FRAC) { s.pinnedCycles = 0; return false; }  // already unpinned

    // Cross the spread to shed `size` contracts in the reducing direction. This
    // only needs to drop |pos| below the pin threshold so two-sided quoting can
    // resume — not to flat. The deliberate "pay a small taker cost to shed risk".
    const reduceSide: Side = pos > 0 ? "ask" : "bid";
    const touch = reduceSide === "ask" ? q.bid : q.ask;
    if (!touch) return false;   // one-sided book: can't cross this cycle; deadline backstops
    const px = reduceSide === "ask"
      ? fp(Math.max(Number(touch) - 0.02, 0.01))     // sell through the bid
      : fp(Math.min(Number(touch) + 0.02, 0.99));    // buy through the ask
    const n = Math.min(s.size, Math.abs(pos));

    // Cancel our resting quotes first so the shed doesn't fight our own orders.
    for (const o of await this.myOrders(k, s.ticker)) await k.cancel(o.order_id ?? o.id).catch(() => {});
    s.bid = s.ask = null;

    console.log(JSON.stringify({ evt: "unpin_shed", ticker: s.ticker, position: pos,
      side: reduceSide, n, px, attempt: (s.shedAttempts ?? 0) + 1 }));
    // post_only=false → taker, crosses. Placed into the tracked `shed` slot so
    // its fills move the inferred position (detectFills, next cycle) and any
    // unfilled remainder is cancelled rather than left resting. Fills are booked
    // at the limit price — conservative, since a cross fills at the touch or better.
    await this.place(k, s, "shed", reduceSide, px, n, false);
    s.shedAttempts = (s.shedAttempts ?? 0) + 1;
    s.pinnedCycles = 0;   // reset patience; subsequent cycles detect if the shed unpinned us
    return true;
  }

  // --- stop + VERIFIED flatten (manual stop, deadline, kill, pin-escalate) --
  // Cancels this market's quotes, then hands off to flattenStep, which closes
  // the exchange's AUTHORITATIVE position. The old doStop flattened the INFERRED
  // s.position once and never checked: stale after an unpin shed or a missed
  // fill (over-/under-flatten), stale by a cycle on the deadline path (which
  // runs before detectFills), and an unfilled close was simply abandoned.
  private async doStop(s: State, reason: string): Promise<void> {
    const k = await this.kalshi();
    s.running = false; s.stopReason = reason;
    s.flattening = true; s.flattenAttempts = 0;
    console.log(`STOP (${reason}) — flattening`);
    try {
      // Cancel ONLY this market's resting orders. The old code cancelled every
      // order on the account, so one market stopping would wipe out every OTHER
      // market's quotes — the cross-DO bug that made multi-market unsafe.
      for (const o of await this.myOrders(k, s.ticker)) await k.cancel(o.order_id ?? o.id).catch(() => {});
      s.bid = s.ask = s.shed = null;
      await this.flattenStep(k, s);
    } catch (e) {
      // Stay in flattening mode; the alarm retries the close next cycle.
      console.log(`stop/flatten error (${s.ticker}): ${(e as Error).message}`);
      await this.ctx.storage.put("state", s);
      await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
    }
  }

  // One verified-flatten attempt. Runs from doStop and then from each alarm
  // while `flattening`: cancel any unfilled remainder of the previous close,
  // re-read position_fp, and either finish (flat), retry (place a fresh
  // marketable close for the TRUE size), or give up loudly after the cap.
  private async flattenStep(k: Kalshi, s: State): Promise<void> {
    for (const o of await this.myOrders(k, s.ticker)) await k.cancel(o.order_id ?? o.id).catch(() => {});
    const pos = await k.positionFp(s.ticker);
    s.authPos = pos;
    s.position = pos;   // inferred → truth, so /status stops showing a stale position after a stop
    if (pos === 0) {
      console.log(JSON.stringify({ evt: "flatten_done", ticker: s.ticker, attempts: s.flattenAttempts }));
      return this.finishStop(s);
    }
    if (s.flattenAttempts >= FLATTEN_MAX_ATTEMPTS) {
      console.log(JSON.stringify({ evt: "flatten_failed", ticker: s.ticker, position: pos,
        attempts: s.flattenAttempts, note: "POSITION LEFT OPEN — flatten manually" }));
      s.stopReason += ` | flatten failed, pos=${pos} left open`;
      return this.finishStop(s);
    }
    s.flattenAttempts++;
    const q = await k.bestQuote(s.ticker);
    let side: Side | null = null, px = "";
    if (pos > 0 && q.bid) { side = "ask"; px = fp(Math.max(Number(q.bid) - 0.03, 0.01)); }
    else if (pos < 0 && q.ask) { side = "bid"; px = fp(Math.min(Number(q.ask) + 0.03, 0.99)); }
    if (side) {
      console.log(`closing pos=${pos} via marketable ${side} ${Math.abs(pos)}@${px} (attempt ${s.flattenAttempts})`);
      await k.restLimit(s.ticker, side, px, Math.abs(pos), false).catch(
        (e) => console.log("!! flatten order FAILED:", (e as Error).message));
    } else {
      console.log(`!! cannot price flatten (one-sided book); pos=${pos}, retrying next cycle`);
    }
    await this.ctx.storage.put("state", s);
    await this.ctx.storage.setAlarm(Date.now() + s.pollSeconds * 1000);
  }

  private async finishStop(s: State): Promise<void> {
    s.flattening = false;
    await this.ctx.storage.deleteAlarm();
    await this.ctx.storage.put("state", s);
    // Drop ourselves from the live-market registry — only now that the stop is
    // FINISHED, so /status and /stop-all keep seeing a market that's still
    // flattening. Also covers self-stops (deadline / kill / pin-escalate).
    await this.env.REGISTRY.getByName("registry").remove(s.ticker).catch(() => {});
  }
}
