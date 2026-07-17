/**
 * src/index.ts — control Worker.
 *
 * A single QuoterDO instance ("main") runs the always-on quoting loop. This
 * Worker is just the token-gated control plane over it:
 *   GET  /balance  — signer smoke test
 *   GET  /status   — current session snapshot
 *   POST /start    — begin quoting {ticker, size?, maxPosition?, maxLoss?, minutes?, pollSeconds?}
 *   POST /stop     — stop + flatten now
 *
 * Every route requires `Authorization: Bearer <CONTROL_TOKEN>`.
 */

import { Kalshi } from "./kalshi";
import type { StartConfig } from "./quoter";

export interface Env {
  QUOTER: DurableObjectNamespace<import("./quoter").QuoterDO>;
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
    const quoter = env.QUOTER.getByName("main");
    try {
      if (url.pathname === "/balance") {
        const k = await Kalshi.create(env.KALSHI_API_KEY_ID, env.KALSHI_PRIVATE_KEY_B64);
        return ok({ balance_dollars: await k.balanceDollars() });
      }
      if (url.pathname === "/status") return ok(await quoter.status());
      if (url.pathname === "/start" && req.method === "POST") {
        const cfg = (await req.json()) as StartConfig;
        if (!cfg?.ticker) return err("ticker required");
        return ok(await quoter.start(cfg));
      }
      if (url.pathname === "/stop" && req.method === "POST") {
        return ok(await quoter.stop());
      }
      return err("not found", 404);
    } catch (e) {
      return err((e as Error).message, 500);
    }
  },
};

export { QuoterDO } from "./quoter";
