"""
src/analysis/model_vs_market.py

THE flagship chart (CLAUDE.md §2, §9): the moneyline model's calibration
curve and the Kalshi market's calibration curve, on the same axes, over
the same games.

Both are evaluated on the intersection of (a) the model's 2026 holdout
predictions and (b) the games that have a settled Kalshi KXMLBGAME market
with a closing quote — an apples-to-apples set, so the two Brier scores
are directly comparable.

Market probability per game = the mid of the closing yes_bid/yes_ask on the
HOME team's KXMLBGAME market (the market that pays $1 iff the home team
wins), taken from the last candle at or before the market's close_time —
the same point-in-time "implied probability when trading stopped" that
src/analysis/calibration.py uses.

The honest question this answers is NOT "does the model beat the market."
It's "is the model as trustworthy as the market, and where does it differ."
For MLB moneyline, matching the market's calibration with no exploitable
edge is a legitimate, defensible result (§2).

Usage:
    ./.venv/bin/python -m src.analysis.model_vs_market
    ./.venv/bin/python -m src.analysis.model_vs_market --bins 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.calibration import brier_score, compute_bins
from src.model.moneyline import (
    fit_and_predict_holdout,
    load_features,
    season_forward_tune,
)
from src.pipeline.db import connect

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# --- market side -------------------------------------------------------------


# Home-team KXMLBGAME PRE-GAME probability + outcome, for 2026 Final games.
#
# `substring(m.ticker from '[^-]+$')` pulls the last dash-delimited token of
# the ticker, which for KXMLBGAME is the team code; matching it to
# home_team_code selects the "home team wins YES" market (each game has two,
# one per team).
#
# CRITICAL: we sample the last candle at or before FIRST PITCH
# (`raw->>'gameDate'`), NOT before m.close_time. A game market's close_time
# is ~3h after first pitch — the moment the game ENDS — by which point the
# price has already converged to ~0/1. Sampling there is lookahead and
# produces an absurd ~0.07 "market Brier" (the corner-concentration
# artifact, CLAUDE.md §2/§8). First pitch is the honest forecast horizon —
# the sports-betting "closing line" analog.
MARKET_SQL = """
SELECT
    l.game_pk,
    (c.yes_bid_close + c.yes_ask_close) / 2      AS mkt_prob,
    CASE WHEN g.is_home_winner THEN 1 ELSE 0 END AS outcome
FROM games g
JOIN market_game_link l ON l.game_pk = g.game_pk
JOIN markets m ON m.ticker = l.ticker
    AND m.series_ticker = 'KXMLBGAME'
    AND m.status = 'finalized'
    AND m.result IN ('yes', 'no')
JOIN LATERAL (
    SELECT yes_bid_close, yes_ask_close
    FROM market_candles
    WHERE ticker = m.ticker
      AND end_period_ts <= (g.raw->>'gameDate')::timestamptz
    ORDER BY end_period_ts DESC
    LIMIT 1
) c ON TRUE
WHERE g.status = 'Final'
  AND g.official_date >= '2026-01-01'
  AND g.is_home_winner IS NOT NULL
  AND substring(m.ticker from '[^-]+$') = g.home_team_code
  AND c.yes_bid_close IS NOT NULL
  AND c.yes_ask_close IS NOT NULL
"""


def fetch_market_probs() -> dict[int, float]:
    """game_pk → market-implied P(home win) at close, for 2026 games."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(MARKET_SQL)
        return {int(gp): float(p) for gp, p, _ in cur.fetchall()}


# --- overlay plot ------------------------------------------------------------


def plot_overlay(model_bins, market_bins, model_brier, market_brier,
                 n: int, out_path: Path,
                 title: str = "Moneyline — Model vs Market Calibration",
                 xlabel: str = "Predicted P(home win)",
                 unit: str = "games") -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6,
            label="Perfect calibration")

    for bins, color, label in (
        (market_bins, "tab:blue",  f"Kalshi market  (Brier {market_brier:.4f})"),
        (model_bins,  "tab:green", f"Model  (Brier {model_brier:.4f})"),
    ):
        if not bins:
            continue
        centers = np.array([b["mean_predicted"] for b in bins])
        emp = np.array([b["empirical"] for b in bins])
        ci_lo = np.array([b["ci_lo"] for b in bins])
        ci_hi = np.array([b["ci_hi"] for b in bins])
        yerr = [np.maximum(emp - ci_lo, 0.0), np.maximum(ci_hi - emp, 0.0)]
        ax.errorbar(centers, emp, yerr=yerr, fmt="o", markersize=6,
                    capsize=3, linewidth=1.3, color=color, ecolor=color,
                    alpha=0.85, label=label)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Realized frequency (empirical)")
    ax.set_title(f"{title}\n2026 holdout, N = {n} {unit} (95% Wilson CI)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- driver ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/features_moneyline.csv")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    # 1. Train the model exactly as src.model.moneyline does, and get its
    #    2026 holdout predictions keyed by game.
    data = load_features(args.features)
    best_C, _ = season_forward_tune(data)
    hold = fit_and_predict_holdout(data, best_C)
    model_by_game = {int(gp): float(p)
                     for gp, p in zip(hold["game_pk_test"], hold["p_test"])}

    # 2. Pull market probabilities and intersect on game_pk. Both series are
    #    then scored on exactly the same games — the fair comparison.
    market_by_game = fetch_market_probs()
    common = sorted(set(model_by_game) & set(market_by_game))

    outcome_by_game = {int(gp): int(o)
                       for gp, o in zip(hold["game_pk_test"], hold["y_test"])}

    model_p = np.array([model_by_game[g] for g in common])
    market_p = np.array([market_by_game[g] for g in common])
    outcomes = np.array([outcome_by_game[g] for g in common])
    n = len(common)

    model_brier = brier_score(model_p, outcomes)
    market_brier = brier_score(market_p, outcomes)

    print(f"\ncommon games (model holdout ∩ Kalshi market): {n}")
    print(f"  home-win rate:            {outcomes.mean():.3f}")
    print(f"\n  {'':16}{'Brier':>9}{'pred std':>11}")
    print(f"  {'-'*36}")
    print(f"  {'Kalshi market':16}{market_brier:>9.4f}{market_p.std():>11.4f}")
    print(f"  {'Model':16}{model_brier:>9.4f}{model_p.std():>11.4f}")
    # pred std is the resolution proxy: a model that separates games has a
    # wide spread of predictions; one stuck at the base rate is narrow.
    print(f"\n  Δ Brier (model − market): {model_brier - market_brier:+.4f}  "
          f"({'model better' if model_brier < market_brier else 'market better'})")
    print(f"  mean |model − market| disagreement: {np.abs(model_p - market_p).mean():.4f}")

    model_bins = compute_bins(model_p, outcomes, args.bins)
    market_bins = compute_bins(market_p, outcomes, args.bins)

    out_path = OUT_DIR / "calibration_model_vs_market.png"
    plot_overlay(model_bins, market_bins, model_brier, market_brier, n, out_path)
    print(f"\n  saved overlay → {out_path}")


if __name__ == "__main__":
    main()
