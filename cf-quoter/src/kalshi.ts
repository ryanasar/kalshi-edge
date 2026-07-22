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

export interface Quote { bid: string | null; ask: string | null; }

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
    if (!r.ok) throw new Error(`${method} ${endpoint} [${r.status}]: ${await r.text()}`);
    return r.json();
  }

  // --- reads --------------------------------------------------------------
  async balanceDollars(): Promise<string> {
    return (await this.json("GET", "/portfolio/balance")).balance_dollars;
  }

  async bestQuote(ticker: string): Promise<Quote> {
    const m = (await this.json("GET", `/markets/${ticker}`)).market ?? {};
    return { bid: m.yes_bid_dollars ?? null, ask: m.yes_ask_dollars ?? null };
  }

  async restingOrders(): Promise<any[]> {
    return (await this.json("GET", "/portfolio/orders?status=resting&limit=100")).orders ?? [];
  }

  async positions(ticker: string): Promise<any[]> {
    const d = await this.json("GET", "/portfolio/positions?limit=100");
    return (d.market_positions ?? []).filter((p: any) => p.ticker === ticker);
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
  async fills(ticker: string): Promise<any[]> {
    const out: any[] = [];
    let cursor = "";
    for (let i = 0; i < 20; i++) {          // safety cap: 20×200 = 4000 fills
      const q = new URLSearchParams({ ticker, limit: "200" });
      if (cursor) q.set("cursor", cursor);
      const d = await this.json("GET", `/portfolio/fills?${q.toString()}`);
      out.push(...(d.fills ?? []));
      cursor = d.cursor ?? "";
      if (!cursor) break;
    }
    return out;
  }

  // --- writes -------------------------------------------------------------
  /** Rest a limit order. side 'bid'=buy YES, 'ask'=sell YES. Returns order_id. */
  async restLimit(
    ticker: string, side: "bid" | "ask", price: string, count: number,
    postOnly = true,
  ): Promise<string> {
    const body = {
      ticker,
      side,
      count: count.toFixed(2),
      price, // already a fixed-point dollar string
      time_in_force: "good_till_canceled",
      self_trade_prevention_type: "taker_at_cross",
      post_only: postOnly,
      client_order_id: crypto.randomUUID(),
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
