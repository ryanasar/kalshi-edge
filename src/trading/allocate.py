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
# toxicity → default presence (fraction of size we hold near best). Stable markets
# we hold best ~always; event/live markets push us off / pin us. Prefer the
# MEASURED atBestPct from a live run when available.
TOX_PRESENCE = {"STABLE": 0.97, "MILD": 0.90, "EVENT": 0.70, "LIVE": 0.45}
CHUNK = 5.0  # $ (≈ contracts, since a two-sided quote reserves ~$1/contract)


@dataclass
class M:
    ticker: str
    root: str          # event-root for correlation grouping
    pool_window: float # $ pool earnable in the REMAINING window
    C: float           # thin-side competition near best
    base_k: float      # $/contract inventory cost, BEFORE concentration
    eps: float
    tox: str
    target: float      # target_size (under-target scaling reference, ~1000)
    oi: float          # open interest — liquidity available to unwind against
    presence: float    # fraction of our size that sits near best (≤1)
    lam: float         # concentration coefficient: k rises with size/OI
    target_scaling: bool  # apply the min(1,(C+S)/target) pool haircut?

    # subsidy now folds in THREE factors the user flagged:
    #  • scaling(S) = min(1,(C+S)/target)  — market-size / target-meeting haircut,
    #    and our own size RAISES it (helps meet target) → not pure saturation.
    #  • presence                          — % of our size actually near price.
    #  • share = S/(S+C)                    — our cut of the near-best liquidity.
    def subsidy(self, S: float) -> float:
        if S + self.C <= 0:
            return 0.0
        share = S / (S + self.C)
        scaling = min(1.0, (self.C + S) / self.target) if self.target_scaling else 1.0
        return self.pool_window * self.eps * scaling * self.presence * share

    # k grows as we become a bigger fraction of open interest: dominating a market
    # means no one to unwind against (the pin) + our own flow moves price on us.
    def k_eff(self, S: float) -> float:
        conc = 1.0 + self.lam * (S / self.oi if self.oi > 0 else 0.0)
        return self.base_k * TOX_K.get(self.tox, 1.5) * conc

    def net(self, S: float) -> float:
        return self.subsidy(S) - self.k_eff(S) * S

    # scaling(S) and k(S) both depend on S, so the closed form breaks — go numeric.
    def marginal(self, S: float) -> float:
        return (self.net(S + CHUNK) - self.net(S)) / CHUNK   # net gain per contract

    def s_star(self) -> float:
        best_S, best_net, S = 0.0, 0.0, 0.0
        while S <= self.target:
            n = self.net(S)
            if n > best_net:
                best_net, best_S = n, S
            S += CHUNK
        return best_S

    def score(self) -> float:
        k0 = self.base_k * TOX_K.get(self.tox, 1.5)
        return self.pool_window / (self.C * k0) if self.C * k0 > 0 else float("inf")


def load(path: str, eps: float, base_k: float, allow_tox: set[str],
         lam: float, target_scaling: bool, presence_override: float | None) -> list[M]:
    out = []
    for c in json.load(open(path)):
        if c.get("toxicity", "MILD") not in allow_tox:
            continue   # hard toxicity gate — a k-penalty can't price a live-event blowup
        # pool available in the remaining window = pool × (hours left / period)
        frac = min(c["hours_left"], c["period_h"]) / c["period_h"] if c["period_h"] else 0.0
        pool_w = c["pool"] * max(frac, 0.0)
        C = max(c["thin_side"], 1.0)   # floor at 1 so share math is finite
        tox = c.get("toxicity", "MILD")
        root = c.get("event") or c["ticker"].split("-")[0]
        # OI: prefer measured; fall back to a big number (no concentration penalty)
        oi = float(c.get("open_interest") or 0) or 1e9
        presence = presence_override if presence_override is not None else TOX_PRESENCE.get(tox, 0.9)
        out.append(M(c["ticker"], root, pool_w, C, base_k, eps, tox,
                     target=float(c.get("target") or 1000.0), oi=oi,
                     presence=presence, lam=lam, target_scaling=target_scaling))
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
    ap.add_argument("--lam", type=float, default=1.0,
                    help="concentration coeff: k scales by (1 + lam·size/OI)")
    ap.add_argument("--target-scaling", action="store_true",
                    help="apply the min(1,(C+S)/target) pool haircut (UNCONFIRMED — run elasticity test first)")
    ap.add_argument("--presence", type=float, default=None,
                    help="override at-best presence (default: by toxicity; use measured atBestPct)")
    args = ap.parse_args()

    allow = {"STABLE", "MILD", "EVENT", "LIVE"} if args.include_toxic else {"STABLE", "MILD"}
    ms = load(args.json, args.eps, args.base_k, allow, args.lam, args.target_scaling, args.presence)
    alloc = allocate(ms, args.budget, args.hurdle, args.group_cap)

    funded = [(m, alloc[m.ticker]) for m in ms if alloc[m.ticker] > 0]
    funded.sort(key=lambda x: -x[0].net(x[1]))

    print(f"OPTIMAL FLEET  (budget ${args.budget:.0f}, ε={args.eps}, base_k=${args.base_k}, "
          f"lam={args.lam}, target_scaling={args.target_scaling}):\n")
    print(f"{'ticker':<28}{'pool_w':>7}{'C':>5}{'pres':>5}{'score':>7}{'S*':>5}{'SIZE':>5}{'netEV':>7}{'tox':>7}")
    tot_net = tot_cap = 0.0
    for m, S in funded:
        tot_net += m.net(S); tot_cap += S
        print(f"{m.ticker:<28.28}{m.pool_window:>7.0f}{m.C:>5.0f}{m.presence:>5.2f}{m.score():>7.0f}"
              f"{m.s_star():>5.0f}{S:>5.0f}{m.net(S):>7.2f}{m.tox:>7}")
    print(f"\nFUNDED: {len(funded)} positions | capital ${tot_cap:.0f} | "
          f"total net EV ${tot_net:.2f} | avg ROI {100*tot_net/tot_cap if tot_cap else 0:.0f}%")

    skipped_star = [m.ticker for m in ms if m.s_star() <= 0]
    if skipped_star:
        print(f"\nskipped (S*≤0, too crowded/toxic for pool): {len(skipped_star)}")
    print("\nε, k, presence, OI are MEASURED — refit from realized payout / inventory P&L / at-best each cycle.")
    print("target_scaling is a HYPOTHESIS — confirm with the elasticity test before trusting it.")


if __name__ == "__main__":
    main()
