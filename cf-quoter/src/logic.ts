/**
 * src/logic.ts — pure decision helpers for QuoterDO. No I/O, no clock, so the
 * parts of the money path that have bitten us can be unit-tested without the
 * exchange (first slice of the decide() core in docs/design/position-correctness.md).
 */

export const TICK = 0.01;

/** Is our order the ONLY size resting at the best price? `levelSize` is the
 *  exchange's size at best (it includes our order). Unknown size → assume we
 *  are alone, which errs toward HOLDING our price (never ratcheting). */
export function aloneAtBest(levelSize: number | null, ourRem: number): boolean {
  if (levelSize == null || !(levelSize > 0)) return true;
  return levelSize <= ourRem + 1e-9;
}

export interface SkewInput {
  position: number;
  maxPosition: number;
  softFrac: number;
  bid: number; ask: number;                       // best quote (includes our own orders)
  bidSize: number | null; askSize: number | null; // size at best
  ourBid: { price: number; rem: number } | null;
  ourAsk: { price: number; rem: number } | null;
}

/**
 * Position-skewed quote prices. Below the soft threshold, join best on both
 * sides. Past it, sit ONE tick better than the best COMPETING price on the
 * reducing side so it fills and unwinds inventory.
 *
 * The best quote includes our own order. The old code improved one tick on
 * whatever was best, so once our improved quote BECAME the best, the next cycle
 * improved on itself, and the reducing side walked a tick per cycle down to
 * bid+1 tick, giving away the spread. If we're already alone at best, we are
 * that one tick ahead of everyone else, so we hold. Clamped so it never crosses.
 */
export function skewPrices(i: SkewInput): { bidPx: number; askPx: number } {
  let bidPx = i.bid, askPx = i.ask;
  if (Math.abs(i.position) < i.maxPosition * i.softFrac) return { bidPx, askPx };
  const at = (p: number, q: number) => Math.abs(p - q) < 1e-9;
  if (i.position > 0) {          // long → cheaper ask, sell down
    const holding = i.ourAsk && at(i.ourAsk.price, i.ask) && aloneAtBest(i.askSize, i.ourAsk.rem);
    if (!holding) askPx = Math.max(i.bid + TICK, i.ask - TICK);
  } else if (i.position < 0) {   // short → richer bid, buy back
    const holding = i.ourBid && at(i.ourBid.price, i.bid) && aloneAtBest(i.bidSize, i.ourBid.rem);
    if (!holding) bidPx = Math.min(i.ask - TICK, i.bid + TICK);
  }
  return { bidPx, askPx };
}

/** Total contracts the exchange says this order has filled, or null if the
 *  record doesn't say. Never inferred from initial−remaining: a CANCELED order
 *  reports remaining 0, which would read as "fully filled". */
export function filledCount(o: any): number | null {
  const f = o.fill_count_fp ?? o.fill_count;
  if (f != null && f !== "") return Number(f);
  if (o.status === "executed") return Number(o.initial_count_fp ?? o.initial_count ?? NaN);
  return null;
}

/**
 * New fills on a leg that has LEFT the resting list. `qty` is what we placed,
 * `rem` what we last saw resting, so (qty − rem) is already counted.
 *   • exchange record found → its true fill count minus what we've counted
 *     (a cancel/expiry with no fills is 0, not a phantom fill);
 *   • record unavailable → fall back to the old assumption (gone ⇒ filled).
 */
export function newFillsOnGoneLeg(leg: { qty?: number; rem: number }, record: any | undefined): number {
  if (record && leg.qty != null) {
    const f = filledCount(record);
    if (f != null && Number.isFinite(f)) return Math.max(0, f - (leg.qty - leg.rem));
  }
  return leg.rem;
}
