"""
src/model/totals.py

Total-runs (over/under) model — CLAUDE.md §5.5, all three layers. Unlike
moneyline (a direct Bernoulli), a totals market asks P(total runs > line),
and each game carries a whole ladder of lines (~3.5 … 16.5). So we predict
a DISTRIBUTION of total runs per game and read the probability off it.

  Layer 1 — estimator:   Ridge regression → μ = E[total runs]. Uses feature
                         LEVELS (both offenses + both starters), NOT home−
                         away differentials: totals are about the run
                         ENVIRONMENT (good offense + weak pitching → high
                         total), not who is better.
  Layer 2 — error dist:  Negative Binomial with per-game mean μ and one
                         shared dispersion r, fit by MLE on the training
                         residual structure. Runs are overdispersed count
                         data (variance 20 vs mean 8.9) — Poisson, which
                         forces variance = mean, would understate the tails.
  Layer 3 — prob engine: P(total > line) = 1 − NegBinCDF(line | μ, r).
                         Same call for every line on every game.

Validation is season-forward, same as moneyline: fit on 2024, tune α on
2025 (RMSE of μ), refit on 2024+2025, predict 2026. The dispersion r is fit
on the 2024+2025 fit set only. Then the probability calibration is measured
against the actual Kalshi KXMLBTOTAL lines, alongside the market.

Park factor and weather are deferred — they're the totals resolution lever
(Coors vs Petco is a huge run-environment swing), analogous to how team
offense was the lever for moneyline.

Usage:
    ./.venv/bin/python -m src.model.totals --features outputs/features_moneyline.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.calibration import brier_score, compute_bins
from src.analysis.model_vs_market import plot_overlay
from src.features.game_features import PER_SIDE_KEYS
from src.pipeline.db import connect

# Feature LEVELS: all home_* then all away_* (18 columns), used as-is (no
# differencing) because total runs depends on the combined environment.
LEVEL_COLUMNS = [f"home_{k}" for k in PER_SIDE_KEYS] + [f"away_{k}" for k in PER_SIDE_KEYS]

# Ridge α grid (regularization strength). Season-forward CV picks one.
ALPHA_GRID = np.logspace(-1, 4, 11)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# --- data loading ------------------------------------------------------------


def _pf(s: str) -> float:
    return float(s) if s not in ("", None) else np.nan


def load_features_totals(path: str, label: str = "total_runs") -> dict:
    """Load feature LEVELS + the runs label (`total_runs` for full game,
    `f5_runs` for first-5). Drops rows with no label."""
    X, y, season, game_pk = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row[label] in ("", None):
                continue
            X.append([_pf(row[c]) for c in LEVEL_COLUMNS])
            y.append(int(row[label]))
            season.append(int(row["official_date"][:4]))
            game_pk.append(int(row["game_pk"]))
    return {
        "X": np.array(X, dtype=float),
        "y": np.array(y, dtype=int),
        "season": np.array(season, dtype=int),
        "game_pk": np.array(game_pk, dtype=int),
    }


# --- Layer 1: estimator ------------------------------------------------------


def build_estimator(alpha: float) -> Pipeline:
    """Impute (train-league-mean) → scale → Ridge on total_runs."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def season_forward_tune(data: dict, alpha_grid=ALPHA_GRID) -> tuple[float, list[dict]]:
    """Fit on 2024, score RMSE on 2025 for each α; pick the lowest-RMSE α."""
    X, y, season = data["X"], data["y"], data["season"]
    tr, va = season == 2024, season == 2025
    results = []
    for a in alpha_grid:
        est = build_estimator(a).fit(X[tr], y[tr])
        pred = est.predict(X[va])
        rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
        results.append({"alpha": float(a), "val_rmse": rmse})
    best = min(results, key=lambda r: r["val_rmse"])
    return best["alpha"], results


def fit_and_predict_holdout(data: dict, alpha: float) -> dict:
    """Refit on 2024+2025, predict μ for both the fit set (needed to fit the
    dispersion) and the 2026 holdout."""
    X, y, season = data["X"], data["y"], data["season"]
    fit_mask, test_mask = season <= 2025, season == 2026
    est = build_estimator(alpha).fit(X[fit_mask], y[fit_mask])
    return {
        "est": est,
        "mu_fit": est.predict(X[fit_mask]),
        "y_fit": y[fit_mask],
        "mu_test": est.predict(X[test_mask]),
        "y_test": y[test_mask],
        "game_pk_test": data["game_pk"][test_mask],
    }


# --- Layer 2: negative-binomial dispersion -----------------------------------


def _nb_p(r: float, mu: np.ndarray) -> np.ndarray:
    """scipy nbinom parameterization: n=r (dispersion), p=r/(r+μ) gives
    mean μ and variance μ + μ²/r."""
    return r / (r + mu)


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """MLE of the shared dispersion r given per-game means μ. Smaller r =
    heavier tails. μ is clipped away from 0 so p stays valid."""
    mu = np.clip(mu, 0.5, None)

    def nll(r: float) -> float:
        return -np.sum(nbinom.logpmf(y, r, _nb_p(r, mu)))

    res = minimize_scalar(nll, bounds=(0.5, 500.0), method="bounded")
    return float(res.x)


# --- Layer 3: probability engine ---------------------------------------------


def p_over(mu: np.ndarray, r: float, line: np.ndarray) -> np.ndarray:
    """P(total runs > line) under NegBin(μ, r). Lines are half-integers, so
    nbinom.cdf(line) = P(X ≤ floor(line)) and 1 − that = P(over)."""
    mu = np.clip(mu, 0.5, None)
    return 1.0 - nbinom.cdf(line, r, _nb_p(r, mu))


# --- market side -------------------------------------------------------------


# Every over/under line for 2026 games, priced at first pitch (same honest
# horizon as the moneyline overlay — never at close/settlement). The series
# (KXMLBTOTAL full game / KXMLBF5TOTAL first-5) is a bound parameter.
TOTALS_MARKET_SQL = """
SELECT
    l.game_pk,
    m.floor_strike                               AS line,
    (c.yes_bid_close + c.yes_ask_close) / 2      AS mkt_prob,
    CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS outcome
FROM games g
JOIN market_game_link l ON l.game_pk = g.game_pk
JOIN markets m ON m.ticker = l.ticker
    AND m.series_ticker = %(series)s
    AND m.status = 'finalized'
    AND m.result IN ('yes', 'no')
    AND m.floor_strike IS NOT NULL
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
  AND c.yes_bid_close IS NOT NULL
  AND c.yes_ask_close IS NOT NULL
"""


def fetch_totals_markets(series: str = "KXMLBTOTAL") -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(TOTALS_MARKET_SQL, {"series": series})
        return [{"game_pk": int(gp), "line": float(ln),
                 "mkt_prob": float(mp), "outcome": int(o)}
                for gp, ln, mp, o in cur.fetchall()]


# --- driver ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/features_moneyline.csv")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--f5", action="store_true",
                        help="First-5-innings totals (KXMLBF5TOTAL / f5_runs)")
    args = parser.parse_args()

    label = "f5_runs" if args.f5 else "total_runs"
    series = "KXMLBF5TOTAL" if args.f5 else "KXMLBTOTAL"
    variant = "First-5 Totals" if args.f5 else "Total Runs"
    out_name = ("calibration_f5totals_vs_market.png" if args.f5
                else "calibration_totals_vs_market.png")

    data = load_features_totals(args.features, label)
    print(f"[{variant}] loaded {len(data['y'])} games "
          f"(2024={int((data['season']==2024).sum())}, "
          f"2025={int((data['season']==2025).sum())}, "
          f"2026={int((data['season']==2026).sum())})")

    # Layer 1 --------------------------------------------------------------
    best_alpha, cv = season_forward_tune(data)
    hold = fit_and_predict_holdout(data, best_alpha)
    mu_test, y_test = hold["mu_test"], hold["y_test"]

    rmse = float(np.sqrt(np.mean((mu_test - y_test) ** 2)))
    base_rmse = float(np.sqrt(np.mean((data["y"][data["season"] <= 2025].mean()
                                       - y_test) ** 2)))
    print("\n=== Layer 1: estimator (season-forward) ===")
    print("      alpha     val_rmse")
    for r in cv:
        star = "  <-- best" if r["alpha"] == best_alpha else ""
        print(f"  {r['alpha']:>9.3f}   {r['val_rmse']:.4f}{star}")
    print(f"\n  2026 holdout: mean total runs pred vs actual")
    print(f"    predicted μ mean: {mu_test.mean():.3f}   actual mean: {y_test.mean():.3f}")
    print(f"    baseline RMSE (predict train mean): {base_rmse:.4f}")
    print(f"    MODEL    RMSE:                      {rmse:.4f}")

    # Layer 2 --------------------------------------------------------------
    r_disp = fit_dispersion(hold["y_fit"], hold["mu_fit"])
    implied_var = mu_test.mean() + mu_test.mean() ** 2 / r_disp
    print("\n=== Layer 2: negative-binomial dispersion ===")
    print(f"  fitted r = {r_disp:.3f}   (implied var at μ̄: {implied_var:.2f}, "
          f"actual label var: {y_test.var():.2f})")

    # Layer 3 + market comparison -----------------------------------------
    mu_by_game = {int(g): float(m) for g, m in zip(hold["game_pk_test"], mu_test)}
    markets = [mk for mk in fetch_totals_markets(series) if mk["game_pk"] in mu_by_game]

    model_p = np.array([p_over(mu_by_game[mk["game_pk"]], r_disp, mk["line"])
                        for mk in markets])
    market_p = np.array([mk["mkt_prob"] for mk in markets])
    outcomes = np.array([mk["outcome"] for mk in markets])

    model_brier = brier_score(model_p, outcomes)
    market_brier = brier_score(market_p, outcomes)

    print("\n=== Layer 3: P(over line) vs market, 2026 holdout ===")
    print(f"  totals markets scored: {len(markets)} "
          f"(over {len(mu_by_game)} games)")
    print(f"  {'':16}{'Brier':>9}{'pred std':>11}")
    print(f"  {'-'*36}")
    print(f"  {'Kalshi market':16}{market_brier:>9.4f}{market_p.std():>11.4f}")
    print(f"  {'Model':16}{model_brier:>9.4f}{model_p.std():>11.4f}")
    print(f"\n  Δ Brier (model − market): {model_brier - market_brier:+.4f}  "
          f"({'model better' if model_brier < market_brier else 'market better'})")

    model_bins = compute_bins(model_p, outcomes, args.bins)
    market_bins = compute_bins(market_p, outcomes, args.bins)
    out_path = OUT_DIR / out_name
    plot_overlay(model_bins, market_bins, model_brier, market_brier,
                 len(markets), out_path,
                 title=f"{variant} — Model vs Market Calibration",
                 xlabel="Predicted P(over line)", unit="markets")
    print(f"\n  saved overlay → {out_path}")


if __name__ == "__main__":
    main()
