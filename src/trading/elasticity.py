"""
src/trading/elasticity.py

Measure the two unknowns the allocator depends on — ε (reward factor) and, above
all, WHETHER the pool is haircut by market-size/target-meeting (the "target-
scaling" hypothesis, factor #1). That hypothesis decides the entire sizing rule:

  • NO target-scaling  → subsidy = pool·ε·S/(S+C), SATURATES → finite optimum S*.
  • YES target-scaling → subsidy ≈ pool·ε·S/target for S+C < 1000, LINEAR in S →
    size up as far as inventory/capital allow (the "scale into bigger positions"
    thesis).

TWO TESTS
---------
1. CROSS-SECTIONAL (works on one cycle): back out the implied ε for each market
   under each model. The model whose ε is CONSISTENT across markets (low spread)
   is the one reality uses. Run-1 is baked in below as the seed dataset.

2. LONGITUDINAL / the real elasticity (needs the SAME market farmed at ≥2 different
   sizes across cycles): payout elasticity e = Δlog(payout)/Δlog(size).
     e ≈ 1  and flat            → LINEAR → target-scaling → go big.
     e < 1 and falling with S   → SATURATING → finite S* → the √ optimum holds.
   Run-1 held size CONSTANT (all 80), so it CANNOT measure this — that's the gap.
   To run it: next cycle, farm a few markets at 2× size, log clean C/presence/OI,
   feed the rows in.

Observations schema (one row per market per cycle):
    {ticker, pool, size, C, presence, payout}   (+ optional target, default 1000)

Usage:
    ./.venv/bin/python -m src.trading.elasticity           # run-1 cross-sectional
    ./.venv/bin/python -m src.trading.elasticity obs.json  # your logged rows
"""

from __future__ import annotations

import argparse
import json
import statistics as st

# Run-1 (2026-07-18), all farmed at resting size 80; C is the deep near-best
# competition (incl. our own 80 → "others" ≈ C−80); presence = realized atBestPct.
RUN1 = [
    {"ticker": "USEDCARCPI", "pool": 750, "size": 80, "C": 119, "presence": 1.00, "payout": 6.74},
    {"ticker": "CPINDEX",    "pool": 600, "size": 80, "C": 76,  "presence": 0.97, "payout": 8.35},
    {"ticker": "USPPIYOY",   "pool": 600, "size": 80, "C": 222, "presence": 0.34, "payout": 5.57},
    {"ticker": "USRETAIL",   "pool": 200, "size": 80, "C": 284, "presence": 1.00, "payout": 3.81},
    {"ticker": "BUILDPERMS", "pool": 300, "size": 80, "C": 8,   "presence": 1.00, "payout": 7.81},
]


def implied_eps(o: dict, target_scaling: bool, target: float) -> float:
    """ε that would reproduce this market's payout under the chosen model."""
    S, C = o["size"], o["C"]
    share = S / (S + C) if S + C > 0 else 0.0
    scaling = min(1.0, (C + S) / target) if target_scaling else 1.0
    denom = o["pool"] * scaling * o["presence"] * share
    return o["payout"] / denom if denom > 0 else float("nan")


def cross_sectional(obs: list[dict], target: float) -> None:
    print(f"{'ticker':<12}{'pool':>6}{'C':>5}{'pres':>6}{'share':>7}{'take%':>7}"
          f"{'ε(no-scale)':>12}{'ε(scaled)':>11}")
    e_no, e_yes = [], []
    for o in obs:
        share = o["size"] / (o["size"] + o["C"])
        en = implied_eps(o, False, target); ey = implied_eps(o, True, target)
        e_no.append(en); e_yes.append(ey)
        print(f"{o['ticker']:<12}{o['pool']:>6}{o['C']:>5}{o['presence']:>6.2f}{share:>7.2f}"
              f"{100*o['payout']/o['pool']:>6.2f}%{en:>12.3f}{ey:>11.3f}")

    def cv(xs):  # coefficient of variation = spread/mean; lower = more consistent
        xs = [x for x in xs if x == x]
        return (st.pstdev(xs) / st.mean(xs)) if xs and st.mean(xs) else float("nan")
    cvn, cvy = cv(e_no), cv(e_yes)
    print(f"\nimplied ε spread (coeff of variation, lower=better fit):")
    print(f"  no target-scaling : mean {st.mean([x for x in e_no if x==x]):.3f}  CV {cvn:.2f}")
    print(f"  WITH target-scaling: mean {st.mean([x for x in e_yes if x==x]):.3f}  CV {cvy:.2f}")
    verdict = ("target-scaling fits better (more consistent ε)" if cvy < cvn * 0.8 else
               "no-scaling fits better" if cvn < cvy * 0.8 else
               "INCONCLUSIVE — ε spread similar both ways; data too noisy to decide")
    print(f"  → {verdict}")


def longitudinal(obs: list[dict], target: float) -> None:
    """Payout elasticity for markets observed at ≥2 sizes — the decisive test."""
    by: dict[str, list[dict]] = {}
    for o in obs:
        by.setdefault(o["ticker"], []).append(o)
    multi = {k: v for k, v in by.items() if len({o["size"] for o in v}) >= 2}
    if not multi:
        print("\nLONGITUDINAL: no market has ≥2 distinct sizes yet — CANNOT measure")
        print("elasticity. Run-1 held size constant. To measure: farm a few markets")
        print("at 2× size next cycle, log {ticker,pool,size,C,presence,payout}, re-run.")
        return
    import math
    print(f"\nLONGITUDINAL elasticity (e≈1 flat → LINEAR/go-big; e<1 falling → SATURATING):")
    for tk, rows in multi.items():
        rows = sorted(rows, key=lambda r: r["size"])
        a, b = rows[0], rows[-1]
        e = (math.log(b["payout"] / a["payout"]) / math.log(b["size"] / a["size"])
             if a["payout"] > 0 and a["size"] > 0 else float("nan"))
        print(f"  {tk:<12} size {a['size']}→{b['size']}  payout {a['payout']}→{b['payout']}  e={e:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", nargs="?", help="observations JSON (default: baked-in run-1)")
    ap.add_argument("--target", type=float, default=1000.0)
    args = ap.parse_args()
    obs = json.load(open(args.json)) if args.json else RUN1
    print(f"OBSERVATIONS: {len(obs)} rows"
          f"{' (run-1 seed)' if not args.json else ''}\n")
    cross_sectional(obs, args.target)
    longitudinal(obs, args.target)


if __name__ == "__main__":
    main()
