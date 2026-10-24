/**
 * src/kalshi.ts — Kalshi auth + REST, ported from the Python auth.py/orders.py.
 *
 * The signer is the load-bearing part (CLAUDE.md §7). Canonical string is
 *   timestamp_ms + METHOD + path   (path EXCLUDES the query string)
 * signed RSA-PSS / SHA-256 with salt length = 32 (the digest length — the same
 * DIGEST_LENGTH that cost hours in Python; MAX_LENGTH produces a valid sig
 * Kalshi rejects). SubtleCrypto's saltLength: 32 is that exact value.
 *
 * The key is imported as PKCS#8 DER (base64 in the secret) — SubtleCrypto can't
 * read PKCS#1, so we re-encoded the PEM once with openssl offline.
 *
 * Prices/counts are fixed-point dollar STRINGS end-to-end (§8) — never floats
 * in an order body. We only parse to Number for the kill-switch mark, where a
 * float is harmless.
 */

const API_PREFIX = "/trade-api/v2";
const HOST = "https://api.elections.kalshi.com";

export async function importPrivateKey(b64Der: string): Promise<CryptoKey> {
  const der = Uint8Array.from(atob(b64Der), (c) => c.charCodeAt(0));
  return crypto.subtle.importKey(
    "pkcs8",
    der.buffer,
    { name: "RSA-PSS", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

/** Top of book. Sizes are the resting size AT the best price (incl. our own
 *  order, if we're there); null when the exchange didn't report them. */
export interface Quote {
  bid: string | null; ask: string | null;
  bidSize: number | null; askSize: number | null;
}

const PAGE = 200;

/** The exchange answered with a non-2xx status. A 4xx is a DEFINITE rejection
 *  (the order was not placed); anything else that throws — a network error, a
 *  timeout, a 5xx — is AMBIGUOUS: the order may or may not exist. */
export class KalshiHttpError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export class Kalshi {
  constructor(private keyId: string, private key: CryptoKey) {}

  static async create(keyId: string, b64Der: string): Promise<Kalshi> {
    return new Kalshi(keyId, await importPrivateKey(b64Der));
  }

  private async sign(tsMs: string, method: string, path: string): Promise<string> {
    const msg = new TextEncoder().encode(
      tsMs + method.toUpperCase() + path.split("?")[0], // query excluded
    );
    const sig = await crypto.subtle.sign(
      { name: "RSA-PSS", saltLength: 32 },
      this.key,
      msg,
    );
    let bin = "";
    for (const b of new Uint8Array(sig)) bin += String.fromCharCode(b);
    return btoa(bin);
  }

  async request(method: string, endpoint: string, body?: unknown): Promise<Response> {
    const path = API_PREFIX + endpoint;
    const ts = Date.now().toString(); // milliseconds — never seconds
    const headers: Record<string, string> = {
      "KALSHI-ACCESS-KEY": this.keyId,
      "KALSHI-ACCESS-TIMESTAMP": ts,
      "KALSHI-ACCESS-SIGNATURE": await this.sign(ts, method, path),
      "Content-Type": "application/json",
    };
    return fetch(HOST + path, {
      method: method.toUpperCase(),
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  private async json(method: string, endpoint: string, body?: unknown): Promise<any> {
    const r = await this.request(method, endpoint, body);
    if (!r.ok) throw new KalshiHttpError(`${method} ${endpoint} [${r.status}]: ${await r.text()}`, r.status);
    return r.json();
  }

  // --- reads --------------------------------------------------------------
  async balanceDollars(): Promise<string> {
    return (await this.json("GET", "/portfolio/balance")).balance_dollars;
  }

  async bestQuote(ticker: string): Promise<Quote> {
    const m = (await this.json("GET", `/markets/${ticker}`)).market ?? {};
    const size = (x: unknown) => (x == null || x === "" ? null : Number(x));
    return {
      bid: m.yes_bid_dollars ?? null, ask: m.yes_ask_dollars ?? null,
      bidSize: size(m.yes_bid_size_fp), askSize: size(m.yes_ask_size_fp),
    };
  }

  /** Follow `cursor` across pages. Returns the rows and whether `maxPages` ran
   *  out with more still available — callers must not mistake a truncated list
   *  for the whole set (a missing order reads as "gone" = a phantom fill). */
  private async paged(endpoint: string, params: Record<string, string>, key: string,
                      maxPages: number): Promise<{ rows: any[]; truncated: boolean }> {
    const rows: any[] = [];
    let cursor = "";
    for (let i = 0; i < maxPages; i++) {
      const q = new URLSearchParams({ ...params, limit: String(PAGE) });
      if (cursor) q.set("cursor", cursor);
      const d = await this.json("GET", `${endpoint}?${q.toString()}`);
      rows.push(...(d[key] ?? []));
      cursor = d.cursor ?? "";
      if (!cursor) return { rows, truncated: false };
    }
    return { rows, truncated: true };
  }

  /** Resting orders, paginated — scoped to one market when `ticker` is given.
   *  (The old single `limit=100` page was account-wide: past 100 resting orders
   *  our legs fell off the list and detectFills booked them as fills.) */
  async restingOrders(ticker?: string): Promise<any[]> {
    const params: Record<string, string> = { status: "resting" };
    if (ticker) params.ticker = ticker;
    const { rows, truncated } = await this.paged("/portfolio/orders", params, "orders", 10);
    if (truncated) throw new Error(`resting-order list truncated (${rows.length}+)`);
    return rows;
  }

  /** Recent orders for a market in ANY status (resting/executed/canceled), so a
   *  leg that left the resting list can be resolved by what actually happened
   *  to it instead of assumed filled. Newest first; a few pages is plenty. */
  async orders(ticker: string): Promise<any[]> {
    return (await this.paged("/portfolio/orders", { ticker }, "orders", 5)).rows;
  }

  async positions(ticker: string): Promise<any[]> {
    const { rows } = await this.paged("/portfolio/positions", { ticker }, "market_positions", 5);
    return rows.filter((p: any) => p.ticker === ticker);
  }

  /** SHADOW: the exchange's authoritative net YES position for a market. Read the
   *  `position_fp` STRING (§8) — `position` is lossy and doesn't exist here. */
  async positionFp(ticker: string): Promise<number> {
    const ps = await this.positions(ticker);
    return ps.length ? Number(ps[0].position_fp ?? 0) : 0;
  }

  /** SHADOW: every fill for a market (paginated via cursor). The fills feed is
   *  the authoritative source of truth for reconciliation; each fill carries a
   *  unique `fill_id` (the exactly-once idempotency key). */
  async fills(ticker: string): Promise<{ fills: any[]; truncated: boolean }> {
    // safety cap: 20×200 = 4000 fills. Truncation is REPORTED, not hidden — a
    // partial fold would silently give a wrong shadow position.
    const { rows, truncated } = await this.paged("/portfolio/fills", { ticker }, "fills", 20);
    return { fills: rows, truncated };
  }

  // --- writes -------------------------------------------------------------
  /** Rest a limit order. side 'bid'=buy YES, 'ask'=sell YES. Returns order_id.
   *  `clientOrderId` is the idempotency key: callers that must not double-place
   *  generate it up front and persist it BEFORE calling, so a lost ack can be
   *  resolved by looking the order up by this id instead of placing again. */
  async restLimit(
    ticker: string, side: "bid" | "ask", price: string, count: number,
    postOnly = true, clientOrderId: string = crypto.randomUUID(),
  ): Promise<string> {
    const body = {
      ticker,
      side,
      count: count.toFixed(2),
      price, // already a fixed-point dollar string
      time_in_force: "good_till_canceled",
      self_trade_prevention_type: "taker_at_cross",
      post_only: postOnly,
      client_order_id: clientOrderId,
    };
    const d = await this.json("POST", "/portfolio/events/orders", body);
    return (d.order ?? d).order_id;
  }

  async cancel(orderId: string): Promise<void> {
    // V2 cancel — the V1 /portfolio/orders/{id} path is 410'd now.
    await this.json("DELETE", `/portfolio/events/orders/${orderId}`);
  }
}

/** Canonical 4-decimal dollar string, for comparing/formatting prices. */
export function fp(x: string | number): string {
  return Number(x).toFixed(4);
}
