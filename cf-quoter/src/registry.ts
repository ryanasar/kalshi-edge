/**
 * src/registry.ts — Registry: the one DO that remembers WHICH markets are being
 * farmed right now.
 *
 * Going multi-market means one QuoterDO instance per ticker (getByName(ticker)).
 * But a Worker is stateless, so between HTTP requests nothing remembers the set
 * of live tickers — /status has no way to know which DOs to poll, and /stop-all
 * has nothing to iterate. This tiny DO holds that set (in durable storage, so it
 * survives evictions), and is the single source of truth the control plane and
 * the quoters both write to:
 *   • index.ts adds on /start, removes on /stop.
 *   • QuoterDO removes ITSELF when it stops on its own (deadline / kill-switch),
 *     so a market that auto-stops doesn't linger in the list as a zombie.
 */

import { DurableObject } from "cloudflare:workers";
import type { Env } from "./index";

export class Registry extends DurableObject<Env> {
  private async get(): Promise<string[]> {
    return (await this.ctx.storage.get<string[]>("tickers")) ?? [];
  }

  async add(ticker: string): Promise<string[]> {
    const set = await this.get();
    if (!set.includes(ticker)) {
      set.push(ticker);
      await this.ctx.storage.put("tickers", set);
    }
    return set;
  }

  async remove(ticker: string): Promise<string[]> {
    const next = (await this.get()).filter((t) => t !== ticker);
    await this.ctx.storage.put("tickers", next);
    return next;
  }

  async list(): Promise<string[]> {
    return this.get();
  }
}
