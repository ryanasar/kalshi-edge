"""
src/analysis/calibration.py

Computes the Kalshi market's calibration curve — THE flagship chart of
this project — for every settled MLB Tier 1 market.

Pulls the (market_implied_prob_at_first_pitch, realized_outcome) pair for
each market, bins them by predicted probability, and plots empirical
realized frequency against predicted probability with per-bin Wilson-score
95% CIs. Also computes and prints the Brier score.

NOTE: "at first pitch" is load-bearing. Sampling the price at market close
(≈ game end) is lookahead — the price has already resolved to ~0/1 — and
inflates calibration into a meaningless ~0.05 Brier. See CALIBRATION_SQL.

Note: this measures the MARKET's calibration, not our model's. It answers
the question "is Kalshi's aggregate well-calibrated on MLB markets?"
That's a real result on its own, and it's the reference our Week 2 model
has to beat.

Outputs:
    - outputs/calibration_market.png
    - summary table + Brier score to stdout

Usage:
    ./.venv/bin/python -m src.analysis.calibration
    ./.venv/bin/python -m src.analysis.calibration --bins 20
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# matplotlib.use("Agg") BEFORE importing pyplot — no GUI backend needed,
# we're saving to PNG. Prevents "no display" errors on headless machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline.db import connect

# MLB market series we compute the calibration curve over.
# Kept in sync with CLAUDE.md §3. Starts with moneyline + full/half-game
# totals; player props deferred until moneyline model is proven.
TIER_1_SERIES = (
    "KXMLBGAME",       # Moneyline
    "KXMLBTOTAL",      # Full-game total runs
    "KXMLBF5TOTAL",    # First-5 total runs
    "KXMLBSPREAD",     # Run line
    "KXMLBTEAMTOTAL",  # Per-team total runs
)

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


# --- data pull ---------------------------------------------------------------


CALIBRATION_SQL = """
SELECT
    m.series_ticker,
    (c.yes_bid_close + c.yes_ask_close) / 2   AS mkt_prob,
    CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS outcome,
    c.yes_ask_close - c.yes_bid_close         AS spread
FROM markets m

-- Link each market to its game so we can sample at FIRST PITCH, not at
-- close. A game market's close_time is ~3h after first pitch (i.e. game
-- END), by which point the price has already converged to ~0/1. Sampling
-- there is lookahead and yields an absurd ~0.05-0.07 "market Brier" — the
-- corner-concentration artifact (§2/§8). First pitch (raw->>'gameDate') is
-- the honest pre-game forecast horizon: the sports-betting closing line.
JOIN market_game_link l ON l.ticker = m.ticker
JOIN games g ON g.game_pk = l.game_pk

-- Latest candle at or before first pitch: point-in-time correct
-- "implied probability the moment before the game started."
JOIN LATERAL (
    SELECT yes_bid_close, yes_ask_close
    FROM market_candles
    WHERE ticker = m.ticker
      AND end_period_ts <= (g.raw->>'gameDate')::timestamptz
    ORDER BY end_period_ts DESC
    LIMIT 1
) c ON TRUE

WHERE m.status = 'finalized'
  AND m.result IN ('yes', 'no')
  AND m.series_ticker = ANY(%s)
  AND c.yes_bid_close IS NOT NULL
  AND c.yes_ask_close IS NOT NULL
"""


def fetch_calibration_data(
    max_spread: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (probs, outcomes, series) for the calibration corpus.

    If `max_spread` is set, drops markets whose closing bid/ask spread is
    wider than the threshold — these are effectively "no market" quotes
    where the mid isn't a real probability estimate.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(CALIBRATION_SQL, (list(TIER_1_SERIES),))
        rows = cur.fetchall()

    if not rows:
        raise SystemExit(
            "No calibration data found. Make sure markets + market_candles "
            "are populated (backfill_series + backfill_candles)."
        )

    series = [r[0] for r in rows]
    probs = np.array([float(r[1]) for r in rows])
    outcomes = np.array([int(r[2]) for r in rows])
    spreads = np.array([float(r[3]) for r in rows])

    if max_spread is not None:
        keep = spreads <= max_spread
        dropped = int((~keep).sum())
        if dropped:
            print(
                f"Filtered out {dropped} markets with spread > {max_spread} "
                f"(dead/one-sided books, not real quotes)."
            )
        series = [s for s, k in zip(series, keep) if k]
        probs = probs[keep]
        outcomes = outcomes[keep]

    return probs, outcomes, series


# --- statistics --------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion. z=1.96
    gives approximately a 95% CI. Chosen over the naive normal interval
    because it stays inside [0,1] and doesn't collapse to a point when
    p_hat is 0 or 1 — both of which happen in the corner bins of a
    calibration curve.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier = mean squared error between predicted probability and 0/1
    outcome. Scale: 0 is perfect, 0.25 is what you'd get by predicting
    0.5 for every event. Lower is better.
    """
    return float(np.mean((probs - outcomes) ** 2))


def compute_bins(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int
) -> list[dict]:
    """
    Uniform bins on [0,1]. For each non-empty bin: count of markets, sum
    of YES outcomes, empirical fraction, and Wilson CI. Skips empty bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    result = []
    for i in range(n_bins):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(outcomes[mask].sum())
        p_hat = k / n
        ci_lo, ci_hi = wilson_interval(k, n)
        result.append(
            {
                "bin_index": i,
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "bin_center": float((edges[i] + edges[i + 1]) / 2),
                "mean_predicted": float(probs[mask].mean()),
                "empirical": float(p_hat),
                "n": n,
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
            }
        )
    return result


# --- plot --------------------------------------------------------------------


def plot_calibration(
    bins: list[dict], brier: float, n_total: int, out_path: Path,
    title_suffix: str = "pooled",
    title_prefix: str = "Kalshi Market Calibration",
    point_label: str = "Kalshi market (95% Wilson CI)",
    point_color: str = "tab:blue",
    xlabel: str = "Market implied probability (predicted)",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect-calibration diagonal. Every point above this line = the
    # market was UNDER-confident; below = OVER-confident. This is the
    # single visual you're building the whole project to show.
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6,
            label="Perfect calibration")

    if bins:
        centers = np.array([b["mean_predicted"] for b in bins])
        emp = np.array([b["empirical"] for b in bins])
        ci_lo = np.array([b["ci_lo"] for b in bins])
        ci_hi = np.array([b["ci_hi"] for b in bins])
        ns = np.array([b["n"] for b in bins])

        # Error bars are asymmetric because Wilson intervals don't have
        # to be centered on p_hat when p_hat is near 0 or 1.
        yerr_lo = np.maximum(emp - ci_lo, 0.0)
        yerr_hi = np.maximum(ci_hi - emp, 0.0)

        ax.errorbar(
            centers, emp,
            yerr=[yerr_lo, yerr_hi],
            fmt="o", markersize=7, capsize=4, linewidth=1.5,
            color=point_color, ecolor=point_color, alpha=0.85,
            label=point_label,
        )

        # Per-bin sample counts annotate the points. A bin with n=2 with
        # tight CIs would be suspicious; annotating makes it visible.
        for c, e, n in zip(centers, emp, ns):
            ax.annotate(f"n={n}", xy=(c, e), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, alpha=0.8)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Realized frequency of YES (empirical)")
    ax.set_title(
        f"{title_prefix} — {title_suffix}\n"
        f"N = {n_total}    Brier = {brier:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bins", type=int, default=10,
        help="Number of uniform bins along the predicted-probability axis.",
    )
    parser.add_argument(
        "--max-spread", type=float, default=None,
        help=(
            "Optional filter: drop markets whose closing bid/ask spread is "
            "wider than this (e.g. 0.20). Useful to exclude dead books "
            "whose 0.5 mid isn't a real probability."
        ),
    )
    parser.add_argument(
        "--per-series", action="store_true",
        help=(
            "Produce one calibration curve per series (moneyline, totals, "
            "spread, ...) alongside the pooled one. Different market types "
            "often have different bias structures — mid-game totals bow "
            "differently than moneyline favorites."
        ),
    )
    args = parser.parse_args()

    probs, outcomes, series = fetch_calibration_data(max_spread=args.max_spread)
    n_total = int(len(probs))
    n_yes = int(outcomes.sum())
    series_arr = np.array(series)

    brier = brier_score(probs, outcomes)
    bins = compute_bins(probs, outcomes, args.bins)

    print(f"\n=== POOLED (all Tier 1 series) ===")
    print(f"Corpus: {n_total} markets  ({n_yes} YES, {n_total - n_yes} NO)")
    print(f"Brier score: {brier:.4f}")
    print("\n  bin  bin range     mean_pred   empirical   n   95% CI")
    print("  " + "-" * 60)
    for b in bins:
        print(
            f"  {b['bin_index']:>3}"
            f"  [{b['bin_lo']:.2f},{b['bin_hi']:.2f}]"
            f"  {b['mean_predicted']:>10.4f}"
            f"  {b['empirical']:>10.4f}"
            f"  {b['n']:>4}"
            f"  [{b['ci_lo']:.3f},{b['ci_hi']:.3f}]"
        )

    out_path = OUT_DIR / "calibration_market.png"
    plot_calibration(bins, brier, n_total, out_path, title_suffix="pooled")
    print(f"\nSaved plot to {out_path}")

    if args.per_series:
        # One curve per series that has any data. Small-N series (e.g.,
        # KXMLBF5TOTAL if candles haven't finished backfilling) still get
        # a chart — their Wilson CIs will be wide, which is the honest
        # signal that we're bin-thin.
        for s in sorted(set(series_arr)):
            mask = series_arr == s
            s_probs = probs[mask]
            s_outcomes = outcomes[mask]
            s_n = int(len(s_probs))
            if s_n == 0:
                continue
            s_brier = brier_score(s_probs, s_outcomes)
            s_bins = compute_bins(s_probs, s_outcomes, args.bins)

            print(f"\n=== {s} ({s_n} markets) ===")
            print(f"Brier: {s_brier:.4f}")
            for b in s_bins:
                print(
                    f"  bin [{b['bin_lo']:.2f},{b['bin_hi']:.2f}] "
                    f"pred={b['mean_predicted']:.4f} "
                    f"emp={b['empirical']:.4f} "
                    f"n={b['n']}"
                )

            s_out = OUT_DIR / f"calibration_market_{s}.png"
            plot_calibration(s_bins, s_brier, s_n, s_out, title_suffix=s)
            print(f"  → {s_out}")


if __name__ == "__main__":
    main()
