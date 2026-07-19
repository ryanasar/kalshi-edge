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
 *   • stop()/deadline/kill ALL flatten: cancel resting orders AND close any
 *     inventory with a marketable order priced through the touch.
 *   • start() refuses to begin unless the account is flat (no resting orders,
 *     no position) — never resume on top of unknown risk.
 *   • self-tracks position from its own fills, persisting last-seen remaining so
 *     an alarm retry can't double-count.
 */

import { DurableObject } from "cloudflare:workers";
import { Kalshi, fp } from "./kalshi";
import type { Env } from "./index";

const MAKER_RATE = 0.0175;
// v2 position-skew: once |inventory| reaches this fraction of maxPosition, start
// improving the REDUCING side by one tick so the position works back to flat
// instead of pinning at the cap (§ v2 learnings — the USPPIYOY pin).
const SKEW_SOFT_FRAC = 0.6;
const TICK = 0.01;

interface Leg { id: string; price: string; rem: number; }
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
  // cancels or flat-checks goes through this filter.
  private async myOrders(k: Kalshi, ticker: string): Promise<any[]> {
    return (await k.restingOrders()).filter(
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
      bid: null, ask: null,
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
      position: s.position, fills: s.fills, cycles: s.cycles, atBest: s.atBest,
      atBestPct: s.cycles ? Math.round((100 * s.atBest) / s.cycles) : 0,
      cash: +s.cash.toFixed(4), fees: +s.fees.toFixed(4),
      endsInMin: Math.max(0, Math.round((s.endTime - Date.now()) / 60_000)),
    };
  }

  // --- the loop -----------------------------------------------------------
  async alarm(): Promise<void> {
    const s = await this.ctx.storage.get<State>("state");
    if (!s || !s.running) return;
    const k = await this.kalshi();
    try {
      if (Date.now() >= s.endTime) { await this.doStop(s, "deadline"); return; }
      await this.detectFills(k, s);
      const q = await k.bestQuote(s.ticker);
      if (q.bid && q.ask) {
        const mid = (Number(q.bid) + Number(q.ask)) / 2;
        const pnl = s.cash + s.position * mid - s.fees;
        s.cycles++;
        if (s.bid && s.ask && fp(s.bid.price) === fp(q.bid) && fp(s.ask.price) === fp(q.ask)) s.atBest++;
        if (pnl <= -s.maxLoss) { await this.doStop(s, `killswitch pnl=${pnl.toFixed(2)}`); return; }
        // Position-skew the PRICES. Below the soft threshold both sides sit at
        // best (symmetric — capture spread + subsidy). Past it, improve the
        // reducing side by one tick to become the new best there, so it gets
        // filled and unwinds inventory (a resting quote AT best didn't fill and
        // pinned us last run). The one-tick improve is clamped so it can never
        // cross the book (post_only would reject a cross anyway), so on a 1¢
        // market it simply stays at best. Maker-only — no taker fee.
        let bidPx = Number(q.bid), askPx = Number(q.ask);
        if (Math.abs(s.position) >= s.maxPosition * SKEW_SOFT_FRAC) {
          if (s.position > 0) askPx = Math.max(bidPx + TICK, askPx - TICK);      // long → cheaper ask, sell down
          else if (s.position < 0) bidPx = Math.min(askPx - TICK, bidPx + TICK); // short → richer bid, buy back
        }
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
    for (const o of await k.restingOrders()) live.set(o.order_id ?? o.id, o);
    for (const tag of ["bid", "ask"] as const) {
      const o = s[tag];
      if (!o) continue;
      if (live.has(o.id)) {
        const rem = remaining(live.get(o.id));
        if (rem < o.rem) { this.recordFill(s, tag, o.price, o.rem - rem); o.rem = rem; }
        if (rem <= 0) s[tag] = null;
      } else {
        this.recordFill(s, tag, o.price, o.rem); // gone & we didn't cancel -> filled
        s[tag] = null;
      }
    }
  }

  private recordFill(s: State, tag: "bid" | "ask", price: string, n: number): void {
    if (n <= 0) return;
    const p = Number(price);
    if (tag === "bid") { s.position += n; s.cash -= p * n; }
    else { s.position -= n; s.cash += p * n; }
    s.fees += MAKER_RATE * p * (1 - p) * n;
    s.fills++;
    console.log(`FILL ${tag} ${n}@${price} -> pos=${s.position} cash=${s.cash.toFixed(4)}`);
  }

  // `room` = contracts allowed on this side before hitting the ±maxPosition cap.
  // We quote min(size, room), so even a full fill can't breach the cap — this is
  // the fix for the 53-vs-40 overshoot (the old code checked the cap but still
  // placed a full-size order, so effective cap was maxPosition + size).
  private async requote(k: Kalshi, s: State, tag: "bid" | "ask", room: number, price: string): Promise<void> {
    const want = Math.min(s.size, Math.max(0, room));
    const o = s[tag];
    if (want <= 0) { if (o) { await k.cancel(o.id); s[tag] = null; } return; }
    if (o && fp(o.price) === fp(price) && o.rem <= room) return; // at best & can't breach — keep it
    if (o) { await k.cancel(o.id); s[tag] = null; }
    try {
      const id = await k.restLimit(s.ticker, tag, fp(price), want, true);
      s[tag] = { id, price: fp(price), rem: want };
    } catch (e) {
      console.log(`skip ${tag}@${price}: ${(e as Error).message}`); // post_only rejected a racing quote
    }
  }

  // --- stop + flatten (runs on manual stop, deadline, and kill) -----------
  private async doStop(s: State, reason: string): Promise<void> {
    const k = await this.kalshi();
    s.running = false; s.stopReason = reason;
    console.log(`STOP (${reason}) — flattening`);
    try {
      // Cancel ONLY this market's resting orders. The old code cancelled every
      // order on the account, so one market stopping would wipe out every OTHER
      // market's quotes — the cross-DO bug that made multi-market unsafe.
      for (const o of await this.myOrders(k, s.ticker)) await k.cancel(o.order_id ?? o.id).catch(() => {});
      s.bid = s.ask = null;
      if (s.position !== 0) {
        const q = await k.bestQuote(s.ticker);
        let side: "bid" | "ask" | null = null, px = "";
        if (s.position > 0 && q.bid) { side = "ask"; px = fp(Math.max(Number(q.bid) - 0.03, 0.01)); }
        else if (s.position < 0 && q.ask) { side = "bid"; px = fp(Math.min(Number(q.ask) + 0.03, 0.99)); }
        if (side) {
          console.log(`closing pos=${s.position} via marketable ${side} ${Math.abs(s.position)}@${px}`);
          await k.restLimit(s.ticker, side, px, Math.abs(s.position), false).catch(
            (e) => console.log("!! flatten order FAILED:", (e as Error).message));
        } else {
          console.log(`!! cannot price flatten (one-sided book); pos=${s.position} LEFT OPEN`);
        }
      }
    } finally {
      await this.ctx.storage.deleteAlarm();
      await this.ctx.storage.put("state", s);
      // Drop ourselves from the live-market registry so a self-stop (deadline or
      // kill-switch) doesn't leave a zombie entry that /status keeps polling.
      await this.env.REGISTRY.getByName("registry").remove(s.ticker).catch(() => {});
    }
  }
}
