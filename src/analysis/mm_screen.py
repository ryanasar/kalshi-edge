"""
src/analysis/mm_screen.py

COARSE MARKET-MAKING SCREEN on 1-min candles.

The last open question after §14.2 (no taker edge) and §14.4 (no directional
fade): does resting-maker SPREAD CAPTURE beat ADVERSE SELECTION, net of the
1.75% maker fee? Market-making was the only thread with positive economics,
but §14.4 measured that "moves continue" — i.e. adverse selection is real.
This screens whether the spread income survives it.

This is an OPTIMISTIC-BOUND screen — every assumption leans maker-favorable,
so a NEGATIVE result is decisive (the real, queue-constrained, tick-level
version can only be worse). The optimistic assumptions, stated so we can
defend the bound:
  • Front-of-queue fills: a quote at the touch fills whenever a trade prints
    through its level that minute. Real MMs sit BEHIND existing resting size.
  • Perfect flattening: each fill is marked independently at the mid HORIZON
    minutes later — no inventory risk, no forced one-way accumulation.
  • Quote at the touch every minute (join the best bid / best ask). In a ~1¢
    market that's the only option — can't quote inside a 1-tick spread.

Per-fill P&L attribution (all $/contract, from the MM's side):
  BUY  at p, mid-now m0, mid-future mf:  half_spread = m0 − p,  adverse = mf − m0
  SELL at p:                              half_spread = p − m0,  adverse = m0 − mf
  realized = half_spread + adverse ;  fee = 1.75% · p · (1−p) ;  net = realized − fee
This is exactly §6's "maker net with adverse-selection haircut" — except the
haircut is now MEASURED instead of a 30% guess.

Fill model (causal): quote set at the CLOSE of minute i (bid = yes_bid_close_i,
ask = yes_ask_close_i), filled against minute i+1's trade range (requires
volume>0 — a real aggressor must cross):
    buy  if trade_low(i+1)  ≤ bid_i ;   sell if trade_high(i+1) ≥ ask_i
Mark at the candle nearest ts(i+1) + HORIZON (flatten at game end if beyond).

Usage:
    ./.venv/bin/python -m src.analysis.mm_screen --games 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.db import connect, get_market_bounds
# Reuse the pilot's game selection + event join — same corpus, same labels.
from src.analysis.overreaction_pilot import (
    MAX_SPREAD, WINDOW_HOURS, classify, fetch_events, select_games, _mean,
)

PROD_HOST = CANDIDATE_HOSTS[0]
HORIZON = 5                          # minutes: when the MM marks/flattens a lot
MAKER_RATE = Decimal("0.0175")      # Kalshi maker fee rate
TAKER_RATE = Decimal("0.07")        # for the comparison line only

# Cache the play-by-play per game: a game has many F5-total strikes, and they
# all share one feed — fetch it once.
_EVENT_CACHE: dict[int, list] = {}


def cached_events(game_pk: int) -> list:
    if game_pk not in _EVENT_CACHE:
        _EVENT_CACHE[game_pk] = fetch_events(game_pk)
    return _EVENT_CACHE[game_pk]


def select_liquid_markets(series: str, n: int) -> list[dict]:
    """The most-traded `series` markets on recent Final games — the strikes an
    MM would actually quote. Ranked by summed hourly-candle volume (a liquidity
    proxy). Returns [{game_pk, ticker}]; multiple strikes per game is fine, the
    feed is cached. Same date window as the moneyline screen for comparability."""
    sql = """
        SELECT l.game_pk, l.ticker, COALESCE(SUM(mc.volume), 0) AS vol
        FROM market_game_link l
        JOIN games g   ON g.game_pk = l.game_pk
        JOIN markets m ON m.ticker  = l.ticker AND m.status = 'finalized'
        LEFT JOIN market_candles mc ON mc.ticker = l.ticker AND mc.period_minutes = 60
        WHERE g.status = 'Final'
          AND g.official_date BETWEEN DATE '2026-07-01' AND DATE '2026-07-12'
          AND m.series_ticker = %s
        GROUP BY l.game_pk, l.ticker
        HAVING COALESCE(SUM(mc.volume), 0) > 0
        ORDER BY vol DESC
        LIMIT %s
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (series, n))
        return [{"game_pk": int(gp), "ticker": tk} for gp, tk, _ in cur.fetchall()]


def maker_fee(p: Decimal) -> Decimal:
    """Kalshi fee = rate · C · P · (1−P), C=$1 per contract."""
    return MAKER_RATE * p * (1 - p)


def fetch_candles(client: KalshiClient, ticker: str) -> list[dict]:
    """Full 1-min rows for the last WINDOW_HOURS of a market's life, keeping
    the fields the fill model needs. Same window/quality filter as the pilot."""
    with connect() as conn:
        bounds = get_market_bounds(conn, ticker)
    if bounds is None:
        return []
    _open, close = bounds
    start = int((close - timedelta(hours=WINDOW_HOURS)).timestamp())
    series_ticker = ticker.split("-", 1)[0]
    resp = client.request(
        "GET", f"/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={"start_ts": start, "end_ts": int(close.timestamp()),
                "period_interval": 1},
    )
    resp.raise_for_status()
    rows: list[dict] = []
    for c in resp.json().get("candlesticks", []):
        bid = c.get("yes_bid", {}).get("close_dollars")
        ask = c.get("yes_ask", {}).get("close_dollars")
        if bid is None or ask is None:
            continue
        bid, ask = Decimal(bid), Decimal(ask)
        if bid <= 0 or ask <= 0 or ask >= 1 or (ask - bid) > MAX_SPREAD:
            continue
        pr = c.get("price", {})
        lo = pr.get("low_dollars")
        hi = pr.get("high_dollars")
        rows.append({
            "ts": datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc),
            "mid": (bid + ask) / 2, "bid": bid, "ask": ask,
            "low": Decimal(lo) if lo is not None else None,
            "high": Decimal(hi) if hi is not None else None,
            "vol": Decimal(c.get("volume_fp", "0") or "0"),
        })
    rows.sort(key=lambda r: r["ts"])
    return rows


def mark_mid(mids: list[tuple[datetime, Decimal]], target: datetime) -> Decimal | None:
    """Mid at `target` (nearest candle within 90s); flatten at the last mid if
    target runs past the end of the game."""
    if not mids:
        return None
    if target >= mids[-1][0]:
        return mids[-1][1]
    best, best_dt = None, None
    for ts, m in mids:
        dt = abs((ts - target).total_seconds())
        if best_dt is None or dt < best_dt:
            best, best_dt = m, dt
    return best if (best_dt is not None and best_dt <= 90) else None


def sim_game(client: KalshiClient, g: dict) -> list[dict]:
    """Simulate touch-quoting over one game; return one record per fill."""
    rows = fetch_candles(client, g["ticker"])
    if len(rows) < 10:
        return []
    events = cached_events(g["game_pk"])
    mids = [(r["ts"], r["mid"]) for r in rows]
    fills: list[dict] = []
    for i in range(len(rows) - 1):
        q, nxt = rows[i], rows[i + 1]
        if nxt["vol"] <= 0 or nxt["low"] is None or nxt["high"] is None:
            continue  # no aggressor next minute → nothing can fill
        m0 = q["mid"]
        mf = mark_mid(mids, nxt["ts"] + timedelta(minutes=HORIZON))
        if mf is None:
            continue
        bucket = classify(q["ts"], nxt["ts"], events)  # what drove this minute
        # Each fill is treated as its own round trip: enter here as a maker,
        # exit at the +HORIZON mid. That round trip has TWO fee'd executions,
        # so the honest net subtracts 2·maker_fee. `net1` (one fee) is the
        # optimistic phantom-free-exit number; `net2` is the real one.
        # Resting BID fills if the market traded down to it.
        if nxt["low"] <= q["bid"]:
            p = q["bid"]
            hs, adv, fee = m0 - p, mf - m0, maker_fee(p)   # long after buying
            fills.append({"bucket": bucket, "side": "buy", "half": hs, "adv": adv,
                          "fee": fee, "net1": hs + adv - fee, "net2": hs + adv - 2 * fee})
        # Resting ASK fills if the market traded up to it.
        if nxt["high"] >= q["ask"]:
            p = q["ask"]
            hs, adv, fee = p - m0, m0 - mf, maker_fee(p)   # short after selling
            fills.append({"bucket": bucket, "side": "sell", "half": hs, "adv": adv,
                          "fee": fee, "net1": hs + adv - fee, "net2": hs + adv - 2 * fee})
    return fills


# --- reporting --------------------------------------------------------------

def _c(x: Decimal | None) -> str:
    return f"{float(x)*100:>7.2f}" if x is not None else f"{'—':>7}"


def report(fills: list[dict], n_games: int) -> None:
    print(f"\n{'='*70}\nCOARSE MM SCREEN — {len(fills)} fills over {n_games} games "
          f"(optimistic bound)\n{'='*70}")
    print(f"marking each lot at +{HORIZON}min mid; all values ¢/contract\n")
    print(f"reference costs: maker fee ~0.44¢ · taker fee ~1.75¢ (per side, near mid)\n")
    hdr = (f"{'cohort':>14} {'fills':>6} {'half_sprd':>9} {'adverse':>9} {'fee':>7} "
           f"{'NET¹':>7} {'NET²':>7}")
    print(hdr); print("-" * len(hdr))
    print("(NET¹ = 1 fee, phantom free exit — optimistic; NET² = round-trip, honest)\n")

    def line(label: str, fs: list[dict]) -> None:
        if not fs:
            print(f"{label:>14} {0:>6}"); return
        print(f"{label:>14} {len(fs):>6} {_c(_mean([f['half'] for f in fs]))} "
              f"{_c(_mean([f['adv'] for f in fs]))} {_c(_mean([f['fee'] for f in fs]))} "
              f"{_c(_mean([f['net1'] for f in fs]))} {_c(_mean([f['net2'] for f in fs]))}")

    line("ALL", fills)
    print("-" * len(hdr))
    for b in ("scoring", "news", "unexplained"):
        line(b, [f for f in fills if f["bucket"] == b])
    print("-" * len(hdr))
    for s in ("buy", "sell"):
        line(s, [f for f in fills if f["side"] == s])

    n1, n2 = _mean([f["net1"] for f in fills]), _mean([f["net2"] for f in fills])
    t2 = sum((f["net2"] for f in fills), Decimal(0))
    print(f"\nhonest NET² per fill: {_c(n2).strip()}¢   |   total round-trip P&L: "
          f"${float(t2):.2f} over {n_games} games ({float(t2)/n_games:.3f}/game)")
    print("verdict: MM survives iff NET² > 0 even under these OPTIMISTIC fills")
    print("(front-of-queue, exit-at-mid). Queue-constrained tick reality is worse.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=30,
                    help="moneyline: # games; other series: # liquid markets")
    ap.add_argument("--series", default="KXMLBGAME",
                    help="KXMLBGAME (home moneyline) or e.g. KXMLBF5TOTAL")
    args = ap.parse_args()
    client = KalshiClient.from_env(host=PROD_HOST)
    if args.series == "KXMLBGAME":
        games = select_games(args.games)              # home-contract convention
    else:
        games = select_liquid_markets(args.series, args.games)
    print(f"[{args.series}] selected {len(games)} markets; simulating touch-quoting ...")
    all_fills: list[dict] = []
    for g in games:
        try:
            fs = sim_game(client, g)
        except Exception as e:  # noqa: BLE001
            print(f"  {g['game_pk']}: skipped ({type(e).__name__}: {e})")
            continue
        all_fills.extend(fs)
    report(all_fills, len(games))


if __name__ == "__main__":
    main()
