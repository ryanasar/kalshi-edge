/**
 * src/index.ts — control Worker (MULTI-MARKET).
 *
 * One QuoterDO instance PER TICKER (env.QUOTER.getByName(ticker)) so several
 * markets can be farmed at once, each with its own always-on alarm loop and its
 * own persisted state. A single Registry DO tracks which tickers are live so the
 * aggregate routes below know what to poll / stop. This Worker is just the
 * token-gated control plane over them:
 *   GET  /balance          — signer smoke test (account-wide)
 *   GET  /status           — snapshot of EVERY live market + portfolio totals
 *   GET  /status?ticker=X  — snapshot of one market
 *   POST /start            — begin quoting {ticker, size?, maxPosition?, maxLoss?, minutes?, pollSeconds?}
 *   POST /stop  {ticker}   — stop + flatten ONE market (ticker in body or ?ticker=)
 *   POST /stop-all         — stop + flatten every live market
 *
 * Every route requires `Authorization: Bearer <CONTROL_TOKEN>`.
 */

import { Kalshi } from "./kalshi";
import type { StartConfig } from "./quoter";

export interface Env {
  QUOTER: DurableObjectNamespace<import("./quoter").QuoterDO>;
  REGISTRY: DurableObjectNamespace<import("./registry").Registry>;
  KALSHI_API_KEY_ID: string;
  KALSHI_PRIVATE_KEY_B64: string;
  CONTROL_TOKEN: string;
}

const ok = (b: unknown) => Response.json(b);
const err = (m: string, s = 400) => Response.json({ error: m }, { status: s });

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (req.headers.get("authorization") !== `Bearer ${env.CONTROL_TOKEN}`) {
      return err("unauthorized", 401);
    }
    const registry = env.REGISTRY.getByName("registry");
    const quoter = (ticker: string) => env.QUOTER.getByName(ticker);

    try {
      if (url.pathname === "/balance") {
        const k = await Kalshi.create(env.KALSHI_API_KEY_ID, env.KALSHI_PRIVATE_KEY_B64);
        return ok({ balance_dollars: await k.balanceDollars() });
      }

      if (url.pathname === "/status") {
        const one = url.searchParams.get("ticker");
        if (one) return ok(await quoter(one).status());
        // Aggregate: poll every live market, roll up portfolio-level totals.
        const tickers = await registry.list();
        const markets = (await Promise.all(tickers.map((t) => quoter(t).status()))) as any[];
        const totals = markets.reduce(
          (a, m: any) => ({
            markets: a.markets + 1,
            running: a.running + (m.running ? 1 : 0),
            fills: a.fills + (m.fills ?? 0),
            cash: +(a.cash + (m.cash ?? 0)).toFixed(4),
            fees: +(a.fees + (m.fees ?? 0)).toFixed(4),
          }),
          { markets: 0, running: 0, fills: 0, cash: 0, fees: 0 },
        );
        return ok({ totals, markets });
      }

      if (url.pathname === "/start" && req.method === "POST") {
        const cfg = (await req.json()) as StartConfig;
        if (!cfg?.ticker) return err("ticker required");
        const res = await quoter(cfg.ticker).start(cfg);
        await registry.add(cfg.ticker);
        return ok(res);
      }

      if (url.pathname === "/stop" && req.method === "POST") {
        const body = (await req.json().catch(() => ({}))) as { ticker?: string };
        const ticker = body.ticker ?? url.searchParams.get("ticker") ?? "";
        if (!ticker) return err("ticker required (body.ticker or ?ticker=)");
        const res = await quoter(ticker).stop();
        await registry.remove(ticker);
        return ok(res);
      }

      if (url.pathname === "/stop-all" && req.method === "POST") {
        const tickers = await registry.list();
        const stopped = await Promise.all(
          tickers.map(async (t) => {
            const res = await quoter(t).stop();
            await registry.remove(t);
            return { ticker: t, ...(res as object) };
          }),
        );
        return ok({ stopped });
      }

      return err("not found", 404);
    } catch (e) {
      return err((e as Error).message, 500);
    }
  },
};

export { QuoterDO } from "./quoter";
export { Registry } from "./registry";
