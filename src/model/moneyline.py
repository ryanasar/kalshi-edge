"""
src/model/moneyline.py

Moneyline (home-win) probability model — CLAUDE.md §5.5 Layer 1, v1.

Consumes the feature CSV from `src.features.game_features` and fits an
L2-regularized logistic regression that outputs P(home team wins). For a
binary market the estimator *is* the probability (§5.5: "Bernoulli
directly parameterized by the estimator") — there's no separate Layer-2
error distribution as there is for run totals.

Why logistic, not linear ridge: the target is a 0/1 outcome, so we need a
model whose output is bounded to [0,1] and whose loss is proper for
probabilities (log loss). L2-penalized logistic regression is the
binary-target analog of ridge — same single-knob (C = 1/λ) regularization
philosophy, correct link. Tuned by *season-forward* CV, never random
splits: you can't train on the future.

Pipeline (fit on the TRAIN split only, so nothing leaks):

    SimpleImputer(mean)  raw home_*/away_* → league-average fill for the
                         sparse early-season / debut starters
    FunctionTransformer  10 raw columns → 5 home−away differentials
                         (encodes the prior that only *relative* pitcher
                          quality matters, symmetrically)
    StandardScaler       so the L2 penalty treats k_pct (~0.2) and
                         avg_velo (~92) on equal footing
    LogisticRegression   penalty='l2'; intercept absorbs the ~0.53
                         home-field baseline

Validation protocol:
    tune C:   fit on 2024, score on 2025 (log loss)   → best C
    holdout:  refit on 2024+2025 at best C, predict 2026 (never touched
              during tuning) — the honest number.

Usage:
    ./.venv/bin/python -m src.model.moneyline \\
        --features outputs/features_moneyline.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.analysis.calibration import brier_score, compute_bins, plot_calibration

# Must match src.features.game_features.FEATURE_KEYS and column order.
FEATURE_KEYS = ["k_pct", "bb_pct", "hr_per_9", "xwoba_bip", "avg_velo"]
RAW_COLUMNS = [f"home_{k}" for k in FEATURE_KEYS] + [f"away_{k}" for k in FEATURE_KEYS]
DIFF_NAMES = [f"diff_{k}" for k in FEATURE_KEYS]

# C = 1/λ. Small C = strong regularization. Season-forward CV picks one.
C_GRID = np.logspace(-3, 2, 11)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# --- data loading ------------------------------------------------------------


def _parse_float(s: str) -> float:
    """CSV cell → float, with empty string (a missing feature) → NaN so the
    imputer can fill it downstream."""
    return float(s) if s not in ("", None) else np.nan


def _parse_label(s: str) -> int | None:
    """is_home_winner cell → 1/0, or None for the rare unlabeled tie."""
    if s == "True":
        return 1
    if s == "False":
        return 0
    return None


def load_features(path: str) -> dict:
    """
    Load the feature CSV into arrays. Drops rows with no binary label (the
    one tie in the corpus). Returns X (n,10 raw), y (n,), season (n,),
    game_pk (n,), and the raw date strings.
    """
    X, y, season, game_pk, dates = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = _parse_label(row["is_home_winner"])
            if label is None:
                continue
            X.append([_parse_float(row[c]) for c in RAW_COLUMNS])
            y.append(label)
            season.append(int(row["official_date"][:4]))
            game_pk.append(int(row["game_pk"]))
            dates.append(row["official_date"])

    return {
        "X": np.array(X, dtype=float),
        "y": np.array(y, dtype=int),
        "season": np.array(season, dtype=int),
        "game_pk": np.array(game_pk, dtype=int),
        "dates": dates,
    }


# --- model -------------------------------------------------------------------


def _home_minus_away(X):
    """Collapse the 10 raw columns (home 0-4, away 5-9) into 5 home−away
    differentials. Module-level (not a lambda) so the pipeline stays
    picklable."""
    X = np.asarray(X, dtype=float)
    return X[:, :5] - X[:, 5:]


def build_pipeline(C: float) -> Pipeline:
    """The full leak-free transform + estimator. Every step is fit on the
    training split only when `.fit()` is called."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("diff", FunctionTransformer(_home_minus_away)),
        ("scale", StandardScaler()),
        # L2 is lbfgs's default penalty; passing penalty="l2" explicitly is
        # deprecated in sklearn 1.9, so we rely on the default. Still ridge.
        ("logreg", LogisticRegression(C=C, solver="lbfgs", max_iter=1000)),
    ])


def season_forward_tune(data: dict, C_grid=C_GRID) -> tuple[float, list[dict]]:
    """
    Fit on 2024, score on 2025, for each C. Pick the C with the lowest
    2025 log loss (the proper scoring rule for probabilities). Returns
    (best_C, per-C results).
    """
    X, y, season = data["X"], data["y"], data["season"]
    tr, va = season == 2024, season == 2025

    results = []
    for C in C_grid:
        pipe = build_pipeline(C).fit(X[tr], y[tr])
        p_va = pipe.predict_proba(X[va])[:, 1]
        results.append({
            "C": float(C),
            "val_log_loss": float(log_loss(y[va], p_va)),
            "val_brier": float(brier_score(p_va, y[va])),
        })

    best = min(results, key=lambda r: r["val_log_loss"])
    return best["C"], results


def fit_and_predict_holdout(data: dict, C: float) -> dict:
    """
    Refit on 2024+2025 at the chosen C, predict the 2026 holdout (never
    seen during tuning). Returns predictions, the fitted pipeline, and the
    holdout arrays.
    """
    X, y, season = data["X"], data["y"], data["season"]
    fit_mask = season <= 2025
    test_mask = season == 2026

    pipe = build_pipeline(C).fit(X[fit_mask], y[fit_mask])
    p_test = pipe.predict_proba(X[test_mask])[:, 1]

    return {
        "pipe": pipe,
        "p_test": p_test,
        "y_test": y[test_mask],
        "game_pk_test": data["game_pk"][test_mask],
        "train_home_rate": float(y[fit_mask].mean()),
        "n_fit": int(fit_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


# --- reporting ---------------------------------------------------------------


def _report(data: dict, best_C: float, cv: list[dict], hold: dict,
            n_bins: int) -> None:
    p, y = hold["p_test"], hold["y_test"]
    model_brier = brier_score(p, y)

    # Baseline: predict the train home-win rate for every game. Any real
    # model must beat this trivial constant, or the features add nothing.
    base_rate = hold["train_home_rate"]
    base_brier = brier_score(np.full_like(p, base_rate), y)

    print("\n=== season-forward tuning (train 2024 → validate 2025) ===")
    print("      C        val_log_loss   val_brier")
    for r in cv:
        star = "  <-- best" if r["C"] == best_C else ""
        print(f"  {r['C']:>9.4f}   {r['val_log_loss']:>10.4f}   "
              f"{r['val_brier']:>9.4f}{star}")

    print(f"\n=== 2026 holdout (refit on 2024+2025, C={best_C:.4f}) ===")
    print(f"  fit games:      {hold['n_fit']}")
    print(f"  holdout games:  {hold['n_test']}  "
          f"(home wins {int(y.sum())} = {y.mean():.3f})")
    print(f"  baseline Brier (predict {base_rate:.3f} always): {base_brier:.4f}")
    print(f"  MODEL    Brier:                                  {model_brier:.4f}")
    lift = (base_brier - model_brier) / base_brier * 100
    print(f"  improvement over baseline:                       {lift:+.1f}%")

    # Standardized coefficients — directly comparable in magnitude because
    # StandardScaler put every differential on unit variance. Sign tells
    # direction: a positive coef on diff_k_pct means "home starter misses
    # more bats than away → home more likely to win," as expected.
    logreg = hold["pipe"].named_steps["logreg"]
    print("\n  standardized coefficients (home−away differentials):")
    for name, coef in zip(DIFF_NAMES, logreg.coef_[0]):
        print(f"    {name:<16} {coef:+.4f}")
    # expit(intercept) ≈ home-win prob at a neutral matchup (all diffs 0) —
    # this should land near the league home-field baseline.
    intercept_prob = 1 / (1 + np.exp(-logreg.intercept_[0]))
    print(f"    intercept        {logreg.intercept_[0]:+.4f}  "
          f"(≈ {intercept_prob:.3f} home-win at neutral matchup)")

    bins = compute_bins(p, y, n_bins)
    print("\n  model calibration on 2026 holdout:")
    print("    bin range     mean_pred   empirical   n")
    for b in bins:
        print(f"    [{b['bin_lo']:.2f},{b['bin_hi']:.2f}]"
              f"  {b['mean_predicted']:>10.4f}"
              f"  {b['empirical']:>10.4f}  {b['n']:>4}")

    out_path = OUT_DIR / "calibration_model_moneyline.png"
    plot_calibration(
        bins, model_brier, len(p), out_path,
        title_suffix="2026 holdout",
        title_prefix="Moneyline Model Calibration",
        point_label="Model (95% Wilson CI)",
        point_color="tab:green",
        xlabel="Model predicted P(home win)",
    )
    print(f"\n  saved model calibration curve → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/features_moneyline.csv",
                        help="Path to the feature CSV from game_features")
    parser.add_argument("--bins", type=int, default=10,
                        help="Calibration bins on the predicted-prob axis")
    args = parser.parse_args()

    data = load_features(args.features)
    print(f"loaded {len(data['y'])} labeled games "
          f"(2024={int((data['season']==2024).sum())}, "
          f"2025={int((data['season']==2025).sum())}, "
          f"2026={int((data['season']==2026).sum())})")

    best_C, cv = season_forward_tune(data)
    hold = fit_and_predict_holdout(data, best_C)
    _report(data, best_C, cv, hold, args.bins)


if __name__ == "__main__":
    main()
