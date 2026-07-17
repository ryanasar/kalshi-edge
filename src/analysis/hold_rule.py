"""
src/analysis/hold_rule.py

Test ONE proposed rule end-to-end: "buy YES when the market is under 0.50 but
the model predicts over 0.50, then hold to settlement." (Ryan's idea, 2026-07-15.)

Priced honestly (§6): buy at the first-pitch ASK (cross the spread), hold to
settlement (payoff = $1 iff home wins, else $0), report gross and net of the
Kalshi 7% taker fee. Uses the exact 2026 holdout model predictions from
src.model.moneyline and the same first-pitch market quotes as
model_vs_market.py — no new modeling, just the trade rule applied.

Also runs two references so the 0.50 line can be judged, not assumed:
  • GENERAL +EV rule: buy YES whenever model_p > ask (underpriced by the
    model) — the economically correct version, no 0.50 anchor.
  • Symmetric edge: same but also SELL (buy NO) when model_p < bid.
If the 0.50-crossing rule isn't clearly better than these, the 0.50 anchor
is adding nothing.

Usage:
    ./.venv/bin/python -m src.analysis.hold_rule
"""

from __future__ import annotations

import math

from src.model.moneyline import (
    fit_and_predict_holdout,
    load_features,
    season_forward_tune,
)
from src.pipeline.db import connect

TAKER_FEE = 0.07


def _fee(price: float) -> float:
    """Kalshi taker fee for a 1-lot, rounded UP to the cent (§6 / backtest.py)."""
    return math.ceil(TAKER_FEE * price * (1 - price) * 100) / 100


# First-pitch bid/ask + outcome for 2026 Final home-team KXMLBGAME markets —
# same point-in-time sample as model_vs_market.MARKET_SQL, but keep bid & ask
# (we buy at the ask, not the mid).
MARKET_SQL = """
SELECT l.game_pk, c.yes_bid_close::float8, c.yes_ask_close::float8,
       CASE WHEN g.is_home_winner THEN 1 ELSE 0 END AS outcome
FROM games g
JOIN market_game_link l ON l.game_pk = g.game_pk
JOIN markets m ON m.ticker = l.ticker
    AND m.series_ticker = 'KXMLBGAME' AND m.status = 'finalized'
    AND m.result IN ('yes', 'no')
JOIN LATERAL (
    SELECT yes_bid_close, yes_ask_close FROM market_candles
    WHERE ticker = m.ticker
      AND end_period_ts <= (g.raw->>'gameDate')::timestamptz
    ORDER BY end_period_ts DESC LIMIT 1
) c ON TRUE
WHERE g.status = 'Final' AND g.official_date >= '2026-01-01'
  AND g.is_home_winner IS NOT NULL
  AND substring(m.ticker from '[^-]+$') = g.home_team_code
  AND c.yes_bid_close IS NOT NULL AND c.yes_ask_close IS NOT NULL
"""


def fetch_quotes() -> dict[int, tuple[float, float, int]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(MARKET_SQL)
        return {int(gp): (b, a, o) for gp, b, a, o in cur.fetchall()}


def _summ(name: str, trades: list[tuple[float, int]]) -> None:
    """trades = list of (net_pnl_per_contract, gross_pnl_per_contract)."""
    if not trades:
        print(f"  {name:28} n=0")
        return
    net = [t[0] for t in trades]
    gross = [t[1] for t in trades]
    n = len(trades)
    mean_net = sum(net) / n
    mean_gross = sum(gross) / n
    wins = sum(1 for _, g in trades if g > 0)
    # t-stat of mean net P&L vs 0 (per-trade; games are independent).
    sd = (sum((x - mean_net) ** 2 for x in net) / (n - 1)) ** 0.5 if n > 1 else 0.0
    t = mean_net / (sd / n**0.5) if sd > 0 else 0.0
    print(f"  {name:28} n={n:>4}  win {wins/n:>5.1%}  "
          f"gross {mean_gross:+.4f}  net {mean_net:+.4f}  "
          f"tot ${sum(net)*1:+.2f}  t={t:+.2f}")


def main() -> None:
    data = load_features("outputs/features_moneyline.csv")
    best_C, _ = season_forward_tune(data)
    hold = fit_and_predict_holdout(data, best_C)
    model = {int(g): float(p) for g, p in zip(hold["game_pk_test"], hold["p_test"])}
    quotes = fetch_quotes()
    games = sorted(set(model) & set(quotes))
    print(f"\n2026 holdout ∩ market: {len(games)} games\n")

    the_rule, general, symmetric = [], [], []
    for g in games:
        p = model[g]
        bid, ask, out = quotes[g]

        # THE RULE: market under 0.50 (use mid) but model over 0.50 → buy YES.
        mid = (bid + ask) / 2
        if mid < 0.50 and p > 0.50:
            gross = out - ask
            the_rule.append((gross - _fee(ask), gross))

        # GENERAL +EV: buy YES whenever the model says it's worth more than ask.
        if p > ask:
            gross = out - ask
            general.append((gross - _fee(ask), gross))

        # SYMMETRIC: also short (buy NO at 1−bid) when model < bid.
        if p > ask:
            gross = out - ask
            symmetric.append((gross - _fee(ask), gross))
        elif p < bid:
            gross = (1 - out) - (1 - bid)          # buy NO at (1−bid)
            symmetric.append((gross - _fee(1 - bid), gross))

    print("  rule                          stats")
    print("  " + "-" * 74)
    _summ("THE RULE (mkt<.5<model, buy)", the_rule)
    _summ("general +EV (model>ask, buy)", general)
    _summ("symmetric (buy & short)", symmetric)
    print("\n  gross/net = mean P&L per contract; tot = summed net over all trades.")
    print("  t≈0 ⇒ indistinguishable from zero (no edge). Buy at ASK, hold to settle.\n")


if __name__ == "__main__":
    main()
