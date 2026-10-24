// Scenario tests: the REAL QuoterDO driven alarm-by-alarm against an in-memory
// fake Kalshi exchange (orders, fills, positions, cursor pagination, lost acks,
// taker liquidity). Each "bugN" block reproduces an incident class that FAILS on
// the pre-fix quoter. Run: npm test
import { QuoterDO } from "../src/quoter";
import { skewPrices, newFillsOnGoneLeg, filledCount, aloneAtBest } from "../src/logic";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
  if (!cond) failures++;
}
const r2 = (x: number) => Math.round(x * 100) / 100;

// ---------------- fake exchange ----------------
interface O { order_id: string; client_order_id: string; ticker: string; side: "bid" | "ask";
  price: number; initial: number; filled: number; status: string; }
class Exchange {
  compBid = 0.40; compBidSize = 100; compAsk = 0.50; compAskSize = 100;
  orders: O[] = []; position = 0; fillsLog: any[] = [];
  takerLiquidity = Infinity;
  ackMode: "ok" | "lose-ack" | "fail-before" = "ok";   // applies to the NEXT post only
  posts = 0; PAGE = 1;                                 // page size 1 → every list paginates
  private n = 0;
  resting(t = "T") { return this.orders.filter((o) => o.status === "resting" && o.ticker === t); }
  fill(o: O, k: number, px = o.price) {
    o.filled += k; this.position += o.side === "bid" ? k : -k;
    this.fillsLog.push({ ticker: o.ticker, action: o.side === "bid" ? "buy" : "sell",
      count_fp: k.toFixed(2), yes_price_dollars: px.toFixed(4), fee_cost: "0" });
    if (o.filled >= o.initial) o.status = "executed";
  }
  quote() {
    const bids = [{ p: this.compBid, s: this.compBidSize }, ...this.resting().filter((o) => o.side === "bid").map((o) => ({ p: o.price, s: o.initial - o.filled }))];
    const asks = [{ p: this.compAsk, s: this.compAskSize }, ...this.resting().filter((o) => o.side === "ask").map((o) => ({ p: o.price, s: o.initial - o.filled }))];
    const bb = Math.max(...bids.map((x) => x.p)), ba = Math.min(...asks.map((x) => x.p));
    const eq = (a: number, b: number) => Math.abs(a - b) < 1e-9;
    return { bb, ba, bs: bids.filter((x) => eq(x.p, bb)).reduce((a, x) => a + x.s, 0),
      as: asks.filter((x) => eq(x.p, ba)).reduce((a, x) => a + x.s, 0) };
  }
  json(o: O) {
    return { order_id: o.order_id, client_order_id: o.client_order_id, ticker: o.ticker, status: o.status,
      initial_count_fp: o.initial.toFixed(2), fill_count_fp: o.filled.toFixed(2),
      remaining_count_fp: (o.status === "resting" ? o.initial - o.filled : 0).toFixed(2) };
  }
  page(rows: any[], cursor: string | null, key: string) {
    const start = cursor ? Number(cursor) : 0, end = start + this.PAGE;
    return { [key]: rows.slice(start, end), cursor: end < rows.length ? String(end) : "" };
  }
  async handle(url: string, init: any): Promise<Response> {
    const u = new URL(url); const path = u.pathname.replace("/trade-api/v2", ""); const m = init.method;
    const q = u.searchParams; const J = (b: any, s = 200) => new Response(JSON.stringify(b), { status: s });
    if (path === "/portfolio/balance") return J({ balance_dollars: "1000.0000" });
    if (path.startsWith("/markets/")) {
      const t = this.quote();
      return J({ market: { yes_bid_dollars: t.bb.toFixed(4), yes_ask_dollars: t.ba.toFixed(4),
        yes_bid_size_fp: t.bs.toFixed(2), yes_ask_size_fp: t.as.toFixed(2) } });
    }
    if (path === "/portfolio/orders" && m === "GET") {
      let rows = this.orders.filter((o) => (!q.get("ticker") || o.ticker === q.get("ticker"))
        && (!q.get("status") || o.status === q.get("status")));
      rows = [...rows].reverse();   // newest first
      return J(this.page(rows.map((o) => this.json(o)), q.get("cursor"), "orders"));
    }
    if (path === "/portfolio/events/orders" && m === "POST") {
      const b = JSON.parse(init.body); this.posts++;
      const mode = this.ackMode; this.ackMode = "ok";
      if (mode === "fail-before") throw new TypeError("network error (request never sent)");
      const price = Number(b.price), count = Number(b.count);
      const crosses = b.side === "bid" ? price >= this.compAsk : price <= this.compBid;
      if (crosses && b.post_only) return J({ error: "post only cross" }, 400);
      const o: O = { order_id: `o${++this.n}`, client_order_id: b.client_order_id, ticker: b.ticker,
        side: b.side, price, initial: count, filled: 0, status: "resting" };
      this.orders.push(o);
      if (crosses) {   // taker: fill against competitor liquidity at the competitor price
        const k = Math.min(count, this.takerLiquidity);
        if (k > 0) this.fill(o, k, b.side === "bid" ? this.compAsk : this.compBid);
      }
      if (mode === "lose-ack") throw new TypeError("network error (ack lost)");
      return J({ order: { order_id: o.order_id } });
    }
    if (path.startsWith("/portfolio/events/orders/") && m === "DELETE") {
      const o = this.orders.find((x) => x.order_id === path.split("/").pop());
      if (!o || o.status !== "resting") return J({ error: "not found" }, 404);
      o.status = "canceled"; return J({ order: this.json(o) });
    }
    if (path === "/portfolio/positions")
      return J(this.page(this.position ? [{ ticker: "T", position_fp: this.position.toFixed(2) }] : [], q.get("cursor"), "market_positions"));
    if (path === "/portfolio/fills")
      return J(this.page([...this.fillsLog].reverse(), q.get("cursor"), "fills"));
    return J({ error: `unhandled ${m} ${path}` }, 404);
  }
}

// ---------------- DO harness ----------------
async function newQuoter(ex: Exchange) {
  const kp = await crypto.subtle.generateKey({ name: "RSA-PSS", modulusLength: 2048,
    publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" }, true, ["sign", "verify"]);
  const der = new Uint8Array(await crypto.subtle.exportKey("pkcs8", kp.privateKey));
  let bin = ""; for (const b of der) bin += String.fromCharCode(b);
  (globalThis as any).fetch = (url: string, init: any) => ex.handle(url, init);
  const store = new Map<string, any>(); let alarmAt: number | null = null;
  const ctx = { storage: {
    get: async (k: string) => structuredClone(store.get(k)),
    put: async (k: string, v: any) => { store.set(k, structuredClone(v)); },
    setAlarm: async (t: number) => { alarmAt = t; }, deleteAlarm: async () => { alarmAt = null; },
  } };
  const removed: string[] = [];
  const env = { KALSHI_API_KEY_ID: "kid", KALSHI_PRIVATE_KEY_B64: btoa(bin),
    REGISTRY: { getByName: () => ({ remove: async (t: string) => { removed.push(t); } }) } };
  const q = new QuoterDO(ctx as any, env as any);
  return { q, state: () => store.get("state"), alarmSet: () => alarmAt !== null, removed };
}
const quiet = async <T>(f: () => Promise<T>) => { const l = console.log; console.log = () => {}; try { return await f(); } finally { console.log = l; } };

// ---------------- unit: logic.ts ----------------
{
  const base = { maxPosition: 20, softFrac: 0.6, bid: 0.40, ask: 0.49, bidSize: 100, ourBid: null };
  check("skew: alone at best → hold (no ratchet)",
    skewPrices({ ...base, position: 13, askSize: 15, ourAsk: { price: 0.49, rem: 15 } }).askPx === 0.49);
  check("skew: someone joined our price → improve one tick",
    r2(skewPrices({ ...base, position: 13, askSize: 115, ourAsk: { price: 0.49, rem: 15 } }).askPx) === 0.48);
  check("skew: competitor undercut us → improve one tick on THEM",
    r2(skewPrices({ ...base, position: 13, ask: 0.47, askSize: 100, ourAsk: { price: 0.49, rem: 15 } }).askPx) === 0.46);
  check("skew: below soft threshold → join best",
    skewPrices({ ...base, position: 5, askSize: 100, ourAsk: null }).askPx === 0.49);
  check("skew: 1¢ market never crosses",
    r2(skewPrices({ ...base, position: 13, ask: 0.41, askSize: 100, ourAsk: null }).askPx) === 0.41);
  check("filledCount: canceled order w/ remaining 0 is NOT full", filledCount({ status: "canceled", fill_count_fp: "0.00", remaining_count_fp: "0.00", initial_count_fp: "15.00" }) === 0);
  check("gone leg: exchange cancel → 0 new fills", newFillsOnGoneLeg({ qty: 15, rem: 15 }, { status: "canceled", fill_count_fp: "0.00" }) === 0);
  check("gone leg: partial then cancel → only the unseen part", newFillsOnGoneLeg({ qty: 15, rem: 10 }, { status: "canceled", fill_count_fp: "8.00" }) === 3);
  check("gone leg: no record → legacy assumption", newFillsOnGoneLeg({ qty: 15, rem: 10 }, undefined) === 10);
  check("aloneAtBest: unknown size → hold", aloneAtBest(null, 5));
}

const START = { ticker: "T", size: 15, maxPosition: 20, maxLoss: 1000, pollSeconds: 1, minutes: 60 };

// ---------------- Bug 1: skew ratchet ----------------
{
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START)); await quiet(() => h.q.alarm());
  const bid = ex.resting().find((o) => o.side === "bid")!; ex.fill(bid, 13);   // long 13 ≥ 0.6·20
  const asks: number[] = [];
  for (let i = 0; i < 6; i++) { await quiet(() => h.q.alarm()); asks.push(ex.resting().find((o) => o.side === "ask")!.price); }
  check("bug1: reducing ask improves ONCE then holds", asks.every((p) => r2(p) === 0.49), `asks=${asks.map(r2).join(",")}`);
  ex.compAsk = 0.49;   // a competitor joins our price → we're no longer alone
  await quiet(() => h.q.alarm());
  check("bug1: re-improves when a competitor joins", r2(ex.resting().find((o) => o.side === "ask")!.price) === 0.48);
}

// ---------------- Bug 3: pending leg that fully fills within one cycle ----------------
{
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START));
  ex.ackMode = "lose-ack";                         // first POST (the bid) lands but the ack is lost
  await quiet(() => h.q.alarm());
  check("bug3 setup: bid leg is pending", h.state().bid?.id === "" && !!h.state().bid?.coid);
  const lost = ex.orders.find((o) => o.side === "bid")!; ex.fill(lost, 15);   // fully fills before next poll
  await quiet(() => h.q.alarm());
  check("bug3: fill of an executed pending order is counted", h.state().position === ex.position,
    `inferred=${h.state().position} exchange=${ex.position}`);
  check("bug3: requote respects the real position (room = 20−15 = 5)",
    ex.resting().filter((o) => o.side === "bid").every((o) => o.initial === 5));
}
{
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START));
  ex.ackMode = "fail-before"; await quiet(() => h.q.alarm());
  await quiet(() => h.q.alarm());
  check("bug3: a POST that never landed is re-placed exactly once",
    ex.resting().filter((o) => o.side === "bid").length === 1 && h.state().position === 0);
}

// ---------------- phantom fill: exchange-side cancel ----------------
{
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START)); await quiet(() => h.q.alarm());
  ex.resting().find((o) => o.side === "ask")!.status = "canceled";   // e.g. exchange cancel / STP
  await quiet(() => h.q.alarm());
  check("phantom: exchange cancel is NOT booked as a fill", h.state().position === 0 && ex.position === 0,
    `inferred=${h.state().position}`);
}

// ---------------- Bug 5 + Bug 2: shed tracked & cancelled; stop flattens TRUE position ----------------
{
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start({ ...START, size: 10 })); await quiet(() => h.q.alarm());
  ex.fill(ex.resting().find((o) => o.side === "bid")!, 10); await quiet(() => h.q.alarm());
  ex.fill(ex.resting().find((o) => o.side === "bid")!, 10);           // pinned at +20
  ex.takerLiquidity = 4;                                               // thin book: shed only partly fills
  let shedSeen = false;
  for (let i = 0; i < 12 && !shedSeen; i++) { await quiet(() => h.q.alarm()); shedSeen = !!h.state().shed; }
  check("bug5 setup: unpin shed placed into the tracked slot", shedSeen);
  await quiet(() => h.q.alarm());
  check("bug5: shed's partial fill moves inferred position", h.state().position === ex.position,
    `inferred=${h.state().position} exchange=${ex.position}`);
  check("bug5: shed remainder cancelled, not left resting",
    !ex.orders.some((o) => o.status === "resting" && o.side === "ask" && r2(o.price) === 0.38));

  ex.takerLiquidity = Infinity;
  const r: any = await quiet(() => h.q.stop());
  check("bug2: stop reports flattening until verified", r.flattening === true);
  await quiet(() => h.q.alarm());
  check("bug2: flattened the AUTHORITATIVE position to exactly 0", ex.position === 0, `exchange=${ex.position}`);
  check("bug2: verified stop finishes, alarm cleared, registry removed",
    h.state().flattening === false && !h.alarmSet() && h.removed.includes("T") && h.state().position === 0);
}
{
  // stale inferred position (simulate a missed fill) → old doStop would have flattened the wrong size
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START)); await quiet(() => h.q.alarm());
  ex.position = -7;                                                    // truth the DO never saw
  await quiet(() => h.q.stop()); await quiet(() => h.q.alarm());
  check("bug2: stop closes a position the DO didn't know about", ex.position === 0, `exchange=${ex.position}`);
}
{
  // no liquidity at all → bounded retries, then gives up LOUDLY with nothing left resting
  const ex = new Exchange(); const h = await newQuoter(ex);
  await quiet(() => h.q.start(START)); await quiet(() => h.q.alarm());
  ex.position = 5; ex.takerLiquidity = 0;
  await quiet(() => h.q.stop());
  for (let i = 0; i < 10 && h.state().flattening; i++) await quiet(() => h.q.alarm());
  check("bug2: unfillable flatten gives up after the cap",
    h.state().flattening === false && /left open/.test(h.state().stopReason) && h.state().flattenAttempts === 5);
  check("bug2: no orphaned flatten order left resting", ex.resting().length === 0);
}

console.log(failures ? `\n${failures} FAILED` : "\nALL PASSED");
process.exit(failures ? 1 : 0);
