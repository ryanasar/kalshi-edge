"""
src/analysis/backtest.py

Backtest harness (CLAUDE.md §6, §9 deliverable #4). Takes a probability
model's edge over the Kalshi market and simulates trading it — honestly.

The whole point of this file is to answer the question calibration cannot:
when the model DISAGREES with the market, does betting that disagreement
make money after real trading costs? A model can be perfectly calibrated
and still have no exploitable edge; only a backtest with honest fills tells
you which.

Trade rule (spread is crossed because we fill at the touch, never the mid):
    model_p − yes_ask > θ   →  buy YES at yes_ask
    yes_bid − model_p > θ   →  buy NO  at (1 − yes_bid)
    else                    →  no trade (edge inside the spread)

Three brackets on the SAME trade set (§6 "bracket, don't pick"):
    taker_gross   fill at the touch, no fees           (upper bound)
    taker_net     minus Kalshi 7% taker fee            (the honest headline)
    maker_net     fill at the passive price (earn the spread), 1.75% maker
                  fee, then a ×(1−haircut) adverse-selection haircut, shown
                  separately — a resting quote fills preferentially when the
                  market is moving against it, so raw maker P&L lies.

Fills use first-pitch bid/ask (never close/settlement — that's lookahead,
see calibration.py). Flat 1-contract sizing. Prices are dollars in [0,1].

Usage:
    ./.venv/bin/python -m src.analysis.backtest
    ./.venv/bin/python -m src.analysis.backtest --threshold 0.03
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model import moneyline as ml
from src.model import totals as tot
from src.pipeline.db import connect

FEATURES = "outputs/features_moneyline.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

TAKER_FEE = 0.07
MAKER_FEE = 0.0175
ADVERSE_HAIRCUT = 0.30   # maker P&L multiplied by (1 − this) — see docstring

THRESHOLD_SWEEP = [0.00, 0.02, 0.03, 0.05, 0.08]


# --- fee ---------------------------------------------------------------------


def _fee(rate: float, price: float) -> float:
    """Kalshi fee for a 1-lot: rate·P·(1−P), rounded UP to the cent (Kalshi
    rounds fees up per order). Maximized at P=0.5 (~1.75¢ taker)."""
    return math.ceil(rate * price * (1.0 - price) * 100.0) / 100.0


# --- one market → one (or zero) trade ----------------------------------------


def _trade(m: dict, threshold: float, fee_aware: bool = True,
           max_spread: float | None = None) -> dict | None:
    """
    Decide the trade for one market and return per-bracket P&L, or None if
    the edge doesn't clear the bar.

    m keys: model_prob, yes_bid, yes_ask, outcome (0/1), date.

    fee_aware=True  → the trigger is expected NET edge (gross minus the taker
                      fee we'd pay), so we never take a trade whose entire
                      gross edge is smaller than the fee. This is the
                      principled version of "raise the threshold": it prunes
                      guaranteed net losers instead of guessing a cutoff.
    max_spread      → skip markets wider than this (dead/illiquid books whose
                      taker cost is unwinnable).
    """
    p, bid, ask, y = m["model_prob"], m["yes_bid"], m["yes_ask"], m["outcome"]

    if max_spread is not None and (ask - bid) > max_spread:
        return None

    # Gross edge on each side, then (optionally) net of the taker fee we'd pay.
    yes_edge = p - ask
    no_edge = bid - p
    if fee_aware:
        yes_edge -= _fee(TAKER_FEE, ask)
        no_edge -= _fee(TAKER_FEE, 1.0 - bid)

    if yes_edge > threshold:
        # Long YES. Taker pays the ask; maker rests at the bid (earns spread).
        taker_price, maker_price = ask, bid
        payoff = y                      # YES pays $1 iff outcome == 1
    elif no_edge > threshold:
        # Long NO. Taker pays 1−bid; maker buys NO at 1−ask (rests, earns spread).
        taker_price, maker_price = 1.0 - bid, 1.0 - ask
        payoff = 1 - y                  # NO pays $1 iff outcome == 0
    else:
        return None

    taker_gross = payoff - taker_price
    taker_net = taker_gross - _fee(TAKER_FEE, taker_price)
    maker_net = (payoff - maker_price) - _fee(MAKER_FEE, maker_price)

    return {
        "taker_gross": taker_gross,
        "taker_net":   taker_net,
        "maker_net":   maker_net,      # haircut applied at aggregation
        "cost":        taker_price,    # capital at risk (taker basis)
        "won":         int(payoff > 0.5),
        "date":        m["date"],
    }


# --- aggregate a set of trades into a report ---------------------------------


def summarize(markets: list[dict], threshold: float, fee_aware: bool = True,
              max_spread: float | None = None) -> dict:
    trades = [t for t in (_trade(m, threshold, fee_aware, max_spread)
                          for m in markets) if t]
    n = len(trades)
    if n == 0:
        return {"n_markets": len(markets), "n_bets": 0}

    cost = sum(t["cost"] for t in trades)
    tn = np.array([t["taker_net"] for t in trades])

    # Max drawdown on the date-ordered taker-net equity curve.
    ordered = sorted(trades, key=lambda t: t["date"])
    equity = np.cumsum([t["taker_net"] for t in ordered])
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())

    maker_raw = sum(t["maker_net"] for t in trades)

    return {
        "n_markets":    len(markets),
        "n_bets":       n,
        "cost":         cost,
        "hit_rate":     float(np.mean([t["won"] for t in trades])),
        "taker_gross":  sum(t["taker_gross"] for t in trades),
        "taker_net":    float(tn.sum()),
        "maker_net":    maker_raw,
        "maker_net_hc": maker_raw * (1.0 - ADVERSE_HAIRCUT),
        # t-stat: is per-bet taker-net edge distinguishable from zero?
        "tstat":        float(tn.mean() / tn.std(ddof=1) * math.sqrt(n))
                        if n > 1 and tn.std(ddof=1) > 0 else 0.0,
        "max_drawdown": max_dd,
        "equity_dates": [t["date"] for t in ordered],
        "equity":       equity,
    }


def _print_report(name: str, markets: list[dict], threshold: float,
                  fee_aware: bool = True, max_spread: float | None = None) -> dict:
    s = summarize(markets, threshold, fee_aware, max_spread)
    print(f"\n=== {name}  (edge threshold θ = {threshold:.2f}) ===")
    if s["n_bets"] == 0:
        print(f"  no bets: 0 of {s['n_markets']} markets cleared the threshold")
        return s

    print(f"  bets placed:   {s['n_bets']} of {s['n_markets']} markets")
    print(f"  capital risked: ${s['cost']:.2f}   hit rate: {s['hit_rate']:.3f}")
    print(f"\n  {'bracket':<20}{'P&L':>10}{'ROI':>9}")
    print(f"  {'-'*39}")
    for label, key in (("taker gross", "taker_gross"),
                       ("taker net (headline)", "taker_net"),
                       ("maker net", "maker_net"),
                       ("maker net ×0.70 h/c", "maker_net_hc")):
        pnl = s[key]
        print(f"  {label:<20}{pnl:>+10.2f}{pnl / s['cost'] * 100:>8.1f}%")
    print(f"\n  taker-net t-stat: {s['tstat']:+.2f}   "
          f"max drawdown: ${s['max_drawdown']:.2f}")
    return s


# --- market data providers (train model, attach prices) ----------------------


MONEYLINE_MKT_SQL = """
SELECT l.game_pk, g.official_date, c.yes_bid_close, c.yes_ask_close,
       CASE WHEN g.is_home_winner THEN 1 ELSE 0 END AS outcome
FROM games g
JOIN market_game_link l ON l.game_pk = g.game_pk
JOIN markets m ON m.ticker = l.ticker AND m.series_ticker = 'KXMLBGAME'
    AND m.status = 'finalized' AND m.result IN ('yes', 'no')
JOIN LATERAL (
    SELECT yes_bid_close, yes_ask_close FROM market_candles
    WHERE ticker = m.ticker AND end_period_ts <= (g.raw->>'gameDate')::timestamptz
    ORDER BY end_period_ts DESC LIMIT 1
) c ON TRUE
WHERE g.status = 'Final' AND g.official_date >= '2026-01-01'
  AND g.is_home_winner IS NOT NULL
  AND substring(m.ticker from '[^-]+$') = g.home_team_code
  AND c.yes_bid_close IS NOT NULL AND c.yes_ask_close IS NOT NULL
"""


def moneyline_markets() -> list[dict]:
    data = ml.load_features(FEATURES)
    best_c, _ = ml.season_forward_tune(data)
    hold = ml.fit_and_predict_holdout(data, best_c)
    prob = {int(g): float(p) for g, p in zip(hold["game_pk_test"], hold["p_test"])}

    out = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute(MONEYLINE_MKT_SQL)
        for gp, d, bid, ask, outcome in cur.fetchall():
            if int(gp) in prob:
                out.append({"model_prob": prob[int(gp)], "yes_bid": float(bid),
                            "yes_ask": float(ask), "outcome": int(outcome), "date": d})
    return out


TOTALS_MKT_SQL = """
SELECT l.game_pk, g.official_date, m.floor_strike,
       c.yes_bid_close, c.yes_ask_close,
       CASE WHEN m.result = 'yes' THEN 1 ELSE 0 END AS outcome
FROM games g
JOIN market_game_link l ON l.game_pk = g.game_pk
JOIN markets m ON m.ticker = l.ticker AND m.series_ticker = 'KXMLBTOTAL'
    AND m.status = 'finalized' AND m.result IN ('yes', 'no')
    AND m.floor_strike IS NOT NULL
JOIN LATERAL (
    SELECT yes_bid_close, yes_ask_close FROM market_candles
    WHERE ticker = m.ticker AND end_period_ts <= (g.raw->>'gameDate')::timestamptz
    ORDER BY end_period_ts DESC LIMIT 1
) c ON TRUE
WHERE g.status = 'Final' AND g.official_date >= '2026-01-01'
  AND g.total_runs IS NOT NULL
  AND c.yes_bid_close IS NOT NULL AND c.yes_ask_close IS NOT NULL
"""


def _mu_offsets(window_days: int = 45) -> dict[int, float]:
    """
    PIT season-drift correction for the totals estimator. It's trained on
    2024-25 and systematically under-predicts the higher-scoring 2026
    environment (μ̄ 8.81 vs actual 9.03). For each 2026 game we shift μ by
    (trailing-window league-mean total runs BEFORE that date) − (training
    mean). Uses only games strictly earlier than the game, so it's leak-free
    — exactly how you'd recalibrate online as a season unfolds.
    """
    rows = []
    with open(FEATURES, newline="") as f:
        for r in csv.DictReader(f):
            if r["total_runs"] in ("", None):
                continue
            rows.append((int(r["game_pk"]),
                         date.fromisoformat(r["official_date"]),
                         int(r["total_runs"])))

    train_mean = float(np.mean([tr for _, d, tr in rows if d.year < 2026]))
    dates = np.array([d for _, d, _ in rows])
    totals_arr = np.array([tr for _, _, tr in rows])

    offsets: dict[int, float] = {}
    for gp, d, _ in rows:
        if d.year != 2026:
            continue
        mask = (dates >= d - timedelta(days=window_days)) & (dates < d)
        offsets[gp] = (float(totals_arr[mask].mean()) - train_mean
                       if mask.sum() >= 20 else 0.0)
    return offsets


def totals_markets(recalibrate: bool = True) -> list[dict]:
    data = tot.load_features_totals(FEATURES)
    best_alpha, _ = tot.season_forward_tune(data)
    hold = tot.fit_and_predict_holdout(data, best_alpha)
    r_disp = tot.fit_dispersion(hold["y_fit"], hold["mu_fit"])
    mu = {int(g): float(m) for g, m in zip(hold["game_pk_test"], hold["mu_test"])}

    if recalibrate:
        offsets = _mu_offsets()
        mu = {g: m + offsets.get(g, 0.0) for g, m in mu.items()}

    out = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute(TOTALS_MKT_SQL)
        for gp, d, line, bid, ask, outcome in cur.fetchall():
            if int(gp) in mu:
                p = float(tot.p_over(np.array([mu[int(gp)]]), r_disp,
                                     np.array([float(line)]))[0])
                out.append({"model_prob": p, "yes_bid": float(bid),
                            "yes_ask": float(ask), "outcome": int(outcome), "date": d})
    return out


# --- plots -------------------------------------------------------------------


def plot_equity(summaries: dict[str, dict], threshold: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in summaries.items():
        if s.get("n_bets"):
            ax.plot(s["equity_dates"], s["equity"], linewidth=1.6, label=name)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Game date")
    ax.set_ylabel("Cumulative taker-net P&L ($, 1-lot)")
    ax.set_title(f"Backtest equity curve — taker net of fees (θ={threshold:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- driver ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.00,
                        help="Extra edge threshold on top of the selection rule")
    parser.add_argument("--max-spread", type=float, default=None,
                        help="Skip markets with bid/ask spread wider than this")
    parser.add_argument("--gross-select", action="store_true",
                        help="Select on GROSS edge (old behavior) instead of net-EV")
    parser.add_argument("--no-recal", action="store_true",
                        help="Disable the totals μ season-drift recalibration")
    args = parser.parse_args()

    fee_aware = not args.gross_select
    print("training models and pulling first-pitch prices...")
    print(f"selection: {'NET-EV (fee-aware)' if fee_aware else 'gross edge'}"
          f"   max_spread: {args.max_spread}"
          f"   totals μ recal: {not args.no_recal}")
    books = {"MONEYLINE": moneyline_markets(),
             "TOTALS": totals_markets(recalibrate=not args.no_recal)}

    # Threshold sweep — taker-net P&L/ROI/bets as selectivity rises.
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP  (taker net of fees)")
    print("=" * 60)
    for name, mk in books.items():
        print(f"\n{name}:")
        print(f"  {'θ':>6}{'bets':>7}{'P&L':>10}{'ROI':>9}{'t-stat':>8}")
        for th in THRESHOLD_SWEEP:
            s = summarize(mk, th, fee_aware, args.max_spread)
            if s["n_bets"]:
                print(f"  {th:>6.2f}{s['n_bets']:>7}{s['taker_net']:>+10.2f}"
                      f"{s['taker_net'] / s['cost'] * 100:>8.1f}%{s['tstat']:>+8.2f}")
            else:
                print(f"  {th:>6.2f}{0:>7}{'—':>10}{'—':>9}{'—':>8}")

    # Detailed report + equity curve at the chosen threshold.
    print("\n" + "=" * 60)
    print(f"DETAILED REPORT  (θ = {args.threshold:.2f})")
    print("=" * 60)
    summaries = {name: _print_report(name, mk, args.threshold, fee_aware,
                                     args.max_spread)
                 for name, mk in books.items()}

    out_path = OUT_DIR / "backtest_equity.png"
    plot_equity(summaries, args.threshold, out_path)
    print(f"\nsaved equity curve → {out_path}")


if __name__ == "__main__":
    main()
