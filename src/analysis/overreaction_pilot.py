"""
src/analysis/overreaction_pilot.py

PILOT: does the market OVERREACT to in-game events, and can a news filter
separate the reverting overreactions from the non-reverting real repricings?

This is the go/no-go test for the "fade the overreaction" thesis (see the
trading discussion). It joins two data sources per game, in memory, no
persistence — we prove the effect is real before building ingest for it.

  1. PRICE  — Kalshi 1-minute candles for the game's home-team moneyline
     market (KXMLBGAME-...-{HOME}). Mid = (yes_bid_close + yes_ask_close)/2,
     i.e. the market's P(home win) each minute through the game.
  2. EVENTS — the MLB Stats API /feed/live play-by-play, every play and
     action timestamped to the millisecond (verified: about.startTime /
     endTime present on 100% of plays).

Method (each step is a defensible modeling choice, not an accident):

  • A "move" is a minute where |Δmid| ≥ MOVE_THRESHOLD (default 3¢). These
    are the candidate over/under-reactions worth trading around.

  • Each move is CLASSIFIED by what happened in that minute's window, with
    precedence scoring > news > unexplained:
      - scoring    : the running score changed in the window. A mechanical
                     repricing — the fair value genuinely moved, and the
                     question is only whether the market OVERSHOT it.
      - news       : a fair-value action event (pitching change, ejection,
                     sub) but no score change. A real level shift the price
                     SHOULD hold — fading it is picking pennies in front of
                     a steamroller.
      - unexplained: neither. The move has no cause we can see → there is
                     information we don't have. This is the adverse-selection
                     landmine; the whole thesis is that we must SKIP these.

  • REVERSION is measured at horizons k (default 5 and 10 min). For a move
    from m_prev → m_now, reversion_frac = (m_now − m_{now+k}) / (m_now − m_prev):
        = 1.0  → fully reverted (m returned to m_prev)   [fade wins]
        = 0.0  → stayed put
        < 0    → continued in the move's direction        [fade loses]
    Signed so it reads identically for up-moves and down-moves.

  • FADE CAPTURE (gross, $/contract) = reversion_frac × |move|. This is the
    P&L of taking the opposite side at m_now and unwinding at m_{now+k}.
    We compare its mean to Kalshi round-trip costs near mid:
        taker/taker ≈ 3.5¢   maker/maker ≈ 0.9¢   (7% vs 1.75% of P(1−P))
    A bucket only matters if its gross capture clears the maker round trip.

THE decisive result to look for: scoring overreactions revert (positive
capture) AND unexplained moves do NOT (≤0). If unexplained reverts just as
much, the news filter adds nothing and the edge is illusory.

Usage:
    ./.venv/bin/python -m src.analysis.overreaction_pilot --games 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import requests

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.db import connect, get_market_bounds

PROD_HOST = CANDIDATE_HOSTS[0]
MLB_FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

# --- knobs (all defensible; each is a modeling choice we can move) ----------
MOVE_THRESHOLD = Decimal("0.03")   # 3¢ minute-over-minute = a "move"
WINDOW_HOURS = 5                   # pull the last 5h of candles = the game
HORIZONS = (5, 10)                 # reversion look-ahead, minutes
MAX_SPREAD = Decimal("0.15")       # ignore minutes with no real 2-sided quote
HORIZON_TOL = 90                   # sec: how close a candle must be to T+k

# Action events that genuinely shift fair value → "news" (do NOT fade).
# Deliberately EXCLUDES mound_visit, batter_timeout, game_advisory, stolen
# bases, etc. — those aren't level-shifting information.
NEWS_TYPES = {
    "pitching_substitution",
    "ejection",
    "offensive_substitution",
    "defensive_substitution",
}


# --- data selection ---------------------------------------------------------

def select_games(n: int) -> list[dict]:
    """Recent Final games whose HOME-team moneyline market we can link.
    Recent dates only, so Kalshi still retains their 1-min candles."""
    sql = """
        SELECT g.game_pk, l.ticker, g.home_team_code, g.away_team_code,
               g.official_date, g.home_score, g.away_score
        FROM games g
        JOIN market_game_link l ON l.game_pk = g.game_pk
        WHERE g.status = 'Final'
          AND g.official_date BETWEEN DATE '2026-07-01' AND DATE '2026-07-12'
          AND l.ticker LIKE 'KXMLBGAME-%%'
          AND l.ticker LIKE '%%-' || g.home_team_code       -- home contract only
        ORDER BY g.official_date, g.game_pk
        LIMIT %s
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (n,))
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_mid_series(client: KalshiClient, ticker: str) -> list[tuple[datetime, Decimal]]:
    """1-min mids for the last WINDOW_HOURS of a market's life. Keeps only
    minutes with a real two-sided quote (both sides present, spread sane)."""
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
    out: list[tuple[datetime, Decimal]] = []
    for c in resp.json().get("candlesticks", []):
        bid = c.get("yes_bid", {}).get("close_dollars")
        ask = c.get("yes_ask", {}).get("close_dollars")
        if bid is None or ask is None:
            continue
        bid, ask = Decimal(bid), Decimal(ask)
        if bid <= 0 or ask <= 0 or ask >= 1 or (ask - bid) > MAX_SPREAD:
            continue  # dead/one-sided quote — no tradeable mid
        ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        out.append((ts, (bid + ask) / 2))
    out.sort(key=lambda x: x[0])
    return out


def fetch_events(game_pk: int) -> list[tuple[datetime, str]]:
    """Timestamped (time, bucket) events for a game, bucket ∈ {scoring,news}.
    Scoring = the running total went up on this play (use the play's endTime,
    when the outcome is settled). News = a NEWS_TYPES action event (use its
    startTime, when it hit the wire)."""
    d = requests.get(MLB_FEED.format(pk=game_pk), timeout=30).json()
    plays = d["liveData"]["plays"]["allPlays"]
    events: list[tuple[datetime, str]] = []
    prev_total = 0
    for p in plays:
        r = p.get("result", {})
        total = (r.get("awayScore") or 0) + (r.get("homeScore") or 0)
        if total > prev_total and p["about"].get("endTime"):
            events.append((_iso(p["about"]["endTime"]), "scoring"))
        prev_total = total
        for e in p.get("playEvents", []):
            if e.get("type") == "action" \
               and e.get("details", {}).get("eventType") in NEWS_TYPES \
               and e.get("startTime"):
                events.append((_iso(e["startTime"]), "news"))
    events.sort(key=lambda x: x[0])
    return events


def _iso(s: str) -> datetime:
    """Parse MLB's ...Z ISO timestamp to an aware UTC datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# --- the join + measurement -------------------------------------------------

def classify(lo: datetime, hi: datetime, events: list[tuple[datetime, str]]) -> str:
    """What caused a move whose minute-bar spans (lo, hi]? Precedence
    scoring > news > unexplained."""
    kinds = {k for t, k in events if lo < t <= hi}
    if "scoring" in kinds:
        return "scoring"
    if "news" in kinds:
        return "news"
    return "unexplained"


def mid_at(series: list[tuple[datetime, Decimal]], target: datetime) -> Decimal | None:
    """Mid of the candle closest to `target`, within HORIZON_TOL seconds."""
    best, best_dt = None, None
    for ts, m in series:
        dt = abs((ts - target).total_seconds())
        if best_dt is None or dt < best_dt:
            best, best_dt = m, dt
    return best if (best_dt is not None and best_dt <= HORIZON_TOL) else None


def analyze_game(client: KalshiClient, g: dict) -> list[dict]:
    """Detect moves in one game, classify each, measure reversion at every
    horizon. Returns one record per move."""
    series = fetch_mid_series(client, g["ticker"])
    if len(series) < 10:
        return []
    events = fetch_events(g["game_pk"])
    records: list[dict] = []
    for i in range(1, len(series)):
        (t_prev, m_prev), (t_now, m_now) = series[i - 1], series[i]
        move = m_now - m_prev
        if abs(move) < MOVE_THRESHOLD:
            continue
        rec = {"game_pk": g["game_pk"], "bucket": classify(t_prev, t_now, events),
               "move": move, "m_prev": m_prev}
        for k in HORIZONS:
            m_fut = mid_at(series, t_now + timedelta(minutes=k))
            if m_fut is None:
                rec[f"rev{k}"] = None
                rec[f"cap{k}"] = None
            else:
                frac = (m_now - m_fut) / move          # signed reversion
                rec[f"rev{k}"] = frac
                rec[f"cap{k}"] = frac * abs(move)       # fade P&L, $/contract
        records.append(rec)
    return records


# --- reporting --------------------------------------------------------------

def _mean(xs: list[Decimal]) -> Decimal | None:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _table(records: list[dict], label: str) -> None:
    buckets = ("scoring", "news", "unexplained")
    hdr = f"{'bucket':>12} {'n':>4} {'mean|move|':>10}"
    for k in HORIZONS:
        hdr += f" {'rev@'+str(k):>7} {'cap@'+str(k)+'¢':>8}"
    print(f"\n[{label}]  n={len(records)}")
    print(hdr)
    print("-" * len(hdr))
    for b in buckets:
        rs = [r for r in records if r["bucket"] == b]
        if not rs:
            print(f"{b:>12} {0:>4}")
            continue
        line = f"{b:>12} {len(rs):>4} {float(_mean([abs(r['move']) for r in rs])):>10.3f}"
        for k in HORIZONS:
            rev = _mean([r[f"rev{k}"] for r in rs])
            cap = _mean([r[f"cap{k}"] for r in rs])
            rev_s = f"{float(rev):>7.2f}" if rev is not None else f"{'—':>7}"
            cap_s = f"{float(cap)*100:>8.2f}" if cap is not None else f"{'—':>8}"
            line += f" {rev_s} {cap_s}"
        print(line)


def report(records: list[dict]) -> None:
    print(f"\n{'='*72}\nOVERREACTION PILOT — {len(records)} moves "
          f"(|Δmid| ≥ {float(MOVE_THRESHOLD):.0%}) across games\n{'='*72}")
    print("fade capture is $/contract; beat MAKER round-trip ~$0.009 to matter")
    print("rev@k>0 = reverts (fade wins); rev@k<0 = continues (momentum).")

    _table(records, "ALL moves")

    # Settlement-drift control: when the game is genuinely undecided (mid near
    # 0.5) there is little deterministic march-to-0/1, so any 'continuation'
    # here is real momentum, not the game just resolving. When mid is extreme
    # (<0.2 or >0.8) a move toward the outcome LOOKS like non-reversion purely
    # because the game is ending — that's information accrual, not overreaction.
    undecided = [r for r in records if Decimal("0.30") <= r["m_prev"] <= Decimal("0.70")]
    extreme = [r for r in records if r["m_prev"] < Decimal("0.20") or r["m_prev"] > Decimal("0.80")]
    _table(undecided, "UNDECIDED only (0.30 ≤ mid ≤ 0.70) — minimal settlement drift")
    _table(extreme, "EXTREME only (mid <0.20 or >0.80) — dominated by settlement drift")
    print("\nif UNDECIDED reverts but EXTREME/ALL don't, the negative result was")
    print("settlement drift masking a real overreaction in undecided games.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=30, help="how many games to pilot")
    args = ap.parse_args()

    client = KalshiClient.from_env(host=PROD_HOST)
    games = select_games(args.games)
    print(f"selected {len(games)} games; joining price + play-by-play ...")
    all_records: list[dict] = []
    for g in games:
        try:
            recs = analyze_game(client, g)
        except Exception as e:  # noqa: BLE001 — one bad game shouldn't kill the run
            print(f"  {g['game_pk']} {g['ticker'][-16:]}: skipped ({type(e).__name__}: {e})")
            continue
        print(f"  {g['game_pk']} {g['ticker'][-16:]:>16}  "
              f"{g['away_team_code']}@{g['home_team_code']} "
              f"{g['away_score']}-{g['home_score']}  moves={len(recs)}")
        all_records.extend(recs)
    report(all_records)


if __name__ == "__main__":
    main()
