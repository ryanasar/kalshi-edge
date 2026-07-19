"""
src/trading/allocate.py

Turn a screener candidate list into an OPTIMAL fleet: which markets to farm, at
what size each, and — as a consequence — how many. This is the portfolio layer
on top of incentive_screen.py (which finds candidates) and ev_calc.py (which
explains one market's EV curve).

THE STRATEGY (see the "pure strategy" derivation it implements)
---------------------------------------------------------------
Per market i, net profit for the window is subsidy minus inventory cost:

    net_i(S) = Pool_i · ε · S/(S + C_i)   −   k_i · S
               [ subsidy: SATURATES ]        [ inventory cost:
               [ as share -> 100%   ]          LINEAR in size    ]

  • Pool_i  — reward pool available in the remaining window.
  • ε       — reward-dilution factor: discount × reality. Measured ≈ 0.135
              (0.5 discount × 0.27 realized-vs-model from the first payout).
  • C_i     — competitors resting near best on the thin side (the binding side).
  • k_i     — inventory cost per contract = base_k × toxicity. base_k ≈ $0.05
              anchored on run-1 (~$15 inventory cost over ~290 contracts held).

Because subsidy saturates but cost is linear, each market has a single optimum
(set marginal subsidy = marginal cost):

    S_i* = √(Pool_i · ε · C_i / k_i)  −  C_i        (clamp ≥ 0)

Read-offs: S* grows with √Pool (bigger pools justify bigger size), grows with
1/√k (lower inventory cost → size up everywhere — this is what v2 buys us), and
is ≤ 0 when Pool·ε/k < C (too crowded/toxic for its pool → DON'T PLAY IT).

ALLOCATION (this is where the *number* of positions comes from)
--------------------------------------------------------------
Don't choose N. Water-fill a risk budget by MARGINAL net return: pour each next
risk-dollar into whichever market's next contract earns the most net, and stop
when the best marginal falls below a HURDLE (opportunity cost) or the budget is
spent. The count of funded markets falls out of that curve — scarce capital →
fewer/bigger; ample capital + low hurdle → more/smaller.

    marginal_net_i(S) = ε · Pool_i · C_i / (S + C_i)²  −  k_i

CORRELATION: raw count isn't the risk metric — INDEPENDENT underlyings are. Ten
flight-cancel strikes are one bet (correlated inventory). So we cap total size
per event-root; each strike still earns its own pool, but the group shares one
inventory budget.

ε and k are MEASURED, not sacred — refit them from realized payouts / inventory
P&L each cycle and re-run. This is the exploit step of an online loop.

Usage:
    ./.venv/bin/python -m src.trading.allocate /tmp/screen_w3.json --budget 500
    ./.venv/bin/python -m src.trading.allocate screen.json --eps 0.135 --base-k 0.05 \
        --hurdle 0.02 --group-cap 60
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

# toxicity → inventory-cost multiplier (stable underlyings barely move → tiny k;
# event/live markets reprice violently → k explodes). Mirrors incentive_screen.
TOX_K = {"STABLE": 1.0, "MILD": 1.5, "EVENT": 4.0, "LIVE": 10.0}
CHUNK = 5.0  # $ (≈ contracts, since a two-sided quote reserves ~$1/contract)


@dataclass
class M:
    ticker: str
    root: str          # event-root for correlation grouping
    pool_window: float # $ pool earnable in the REMAINING window
    C: float           # thin-side competition
    k: float           # $/contract inventory cost
    eps: float
    tox: str

    def subsidy(self, S: float) -> float:
        return self.pool_window * self.eps * S / (S + self.C) if (S + self.C) > 0 else 0.0

    def net(self, S: float) -> float:
        return self.subsidy(S) - self.k * S

    def marginal(self, S: float) -> float:
        # d/dS of net = ε·Pool·C/(S+C)² − k
        return self.eps * self.pool_window * self.C / (S + self.C) ** 2 - self.k

    def s_star(self) -> float:
        v = self.eps * self.pool_window * self.C / self.k
        return max(0.0, math.sqrt(v) - self.C)

    def score(self) -> float:
        return self.pool_window / (self.C * self.k) if self.C * self.k > 0 else float("inf")


def load(path: str, eps: float, base_k: float, allow_tox: set[str]) -> list[M]:
    out = []
    for c in json.load(open(path)):
        if c.get("toxicity", "MILD") not in allow_tox:
            continue   # hard toxicity gate — a k-penalty can't price a live-event blowup
        # pool available in the remaining window = pool × (hours left / period)
        frac = min(c["hours_left"], c["period_h"]) / c["period_h"] if c["period_h"] else 0.0
        pool_w = c["pool"] * max(frac, 0.0)
        C = max(c["thin_side"], 1.0)   # floor at 1 so share math is finite
        tox = c.get("toxicity", "MILD")
        k = base_k * TOX_K.get(tox, 1.5)
        root = c.get("event") or c["ticker"].split("-")[0]
        out.append(M(c["ticker"], root, pool_w, C, k, eps, tox))
    return out


def allocate(ms: list[M], budget: float, hurdle: float, group_cap: float) -> dict[str, float]:
    """Greedy marginal-net water-filling with a hurdle + per-root correlation cap.
    Each market is bounded by its own S* (never fund past the optimum)."""
    alloc = {m.ticker: 0.0 for m in ms}
    star = {m.ticker: m.s_star() for m in ms}
    group: dict[str, float] = {}
    spent = 0.0
    while spent + CHUNK <= budget + 1e-9:
        best, best_gain = None, hurdle  # must beat the hurdle to fund
        for m in ms:
            if alloc[m.ticker] + CHUNK > star[m.ticker]:      # past this market's optimum
                continue
            if group.get(m.root, 0.0) + CHUNK > group_cap:    # correlated-risk cap
                continue
            g = m.marginal(alloc[m.ticker])                   # marginal $/contract at current size
            if g > best_gain:
                best_gain, best = g, m
        if best is None:
            break   # nothing clears the hurdle within caps → stop (budget may remain)
        alloc[best.ticker] += CHUNK
        group[best.root] = group.get(best.root, 0.0) + CHUNK
        spent += CHUNK
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", help="screener JSON (from incentive_screen --json)")
    ap.add_argument("--budget", type=float, default=500, help="$ inventory/risk budget")
    ap.add_argument("--eps", type=float, default=0.03,
                    help="reward factor (MEASURE it): run-1 realized payouts imply ~0.03")
    ap.add_argument("--base-k", type=float, default=0.05, help="$/contract inventory cost anchor")
    ap.add_argument("--hurdle", type=float, default=0.01, help="min marginal net $/contract to fund")
    ap.add_argument("--group-cap", type=float, default=80, help="max total size per correlated event-root")
    ap.add_argument("--include-toxic", action="store_true",
                    help="keep EVENT/LIVE markets (default: STABLE+MILD only)")
    args = ap.parse_args()

    allow = {"STABLE", "MILD", "EVENT", "LIVE"} if args.include_toxic else {"STABLE", "MILD"}
    ms = load(args.json, args.eps, args.base_k, allow)
    alloc = allocate(ms, args.budget, args.hurdle, args.group_cap)

    funded = [(m, alloc[m.ticker]) for m in ms if alloc[m.ticker] > 0]
    funded.sort(key=lambda x: -x[0].net(x[1]))

    print(f"OPTIMAL FLEET  (budget ${args.budget:.0f}, ε={args.eps}, base_k=${args.base_k}, "
          f"hurdle ${args.hurdle}/ctr, group-cap {args.group_cap:.0f}):\n")
    print(f"{'ticker':<30}{'pool_w':>7}{'C':>5}{'k':>6}{'score':>7}{'S*':>5}{'SIZE':>5}{'netEV':>7}{'tox':>7}")
    tot_net = tot_cap = 0.0
    for m, S in funded:
        tot_net += m.net(S); tot_cap += S
        print(f"{m.ticker:<30.30}{m.pool_window:>7.0f}{m.C:>5.0f}{m.k:>6.2f}{m.score():>7.0f}"
              f"{m.s_star():>5.0f}{S:>5.0f}{m.net(S):>7.2f}{m.tox:>7}")
    print(f"\nFUNDED: {len(funded)} positions | capital ${tot_cap:.0f} | "
          f"total net EV ${tot_net:.2f} | avg ROI {100*tot_net/tot_cap if tot_cap else 0:.0f}%")

    skipped_star = [m.ticker for m in ms if m.s_star() <= 0]
    if skipped_star:
        print(f"\nskipped (S*≤0, too crowded/toxic for pool): {len(skipped_star)} "
              f"e.g. {', '.join(skipped_star[:5])}")
    print("\nε and k are MEASURED — refit from realized payout / inventory P&L each cycle.")


if __name__ == "__main__":
    main()
