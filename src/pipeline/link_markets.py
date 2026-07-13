"""
src/pipeline/link_markets.py

Populate `market_game_link` by parsing Kalshi tickers and looking up the
corresponding MLB `game_pk` from the `games` table.

Ticker format (verified against the settled corpus):

    K{SERIES}-{YY}{MMM}{DD}{HHMM}{AWAY}{HOME}[G{N}]-[{SUFFIX}]

    e.g.  KXMLBGAME-26JUL111605MILPIT-PIT             (moneyline, home suffix)
          KXMLBTOTAL-26MAY181840CLEDET-11             (totals, threshold suffix)
          KXMLBSPREAD-26MAY311610PHILAD-LAD10         (spread, team+threshold suffix)
          KXMLBF5TOTAL-26MAY291910MIANYM-2            (F5 totals, threshold suffix)
          KXMLBTEAMTOTAL-26MAY091810MINCLE-MIN5       (team total, team+threshold)

The suffix format varies by series (team code, integer, team+integer),
so we do NOT use it to identify teams. The middle string (AWAY concat
HOME, possibly with a trailing G{N} for doubleheaders) is split against
the known 30-team abbreviation set. No team code is a prefix of another,
so the split is deterministic.

Time (HHMM) and game_number are used only for doubleheader tie-breaks —
most single-game dates have exactly one matching game_pk and don't need
either.

Usage:
    # link every unlinked market
    ./.venv/bin/python -m src.pipeline.link_markets

    # only one series
    ./.venv/bin/python -m src.pipeline.link_markets --series KXMLBGAME

    # dry run (report matches without writing)
    ./.venv/bin/python -m src.pipeline.link_markets --dry-run
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date

import requests

from src.pipeline.db import (
    connect,
    insert_market_game_link,
    list_unlinked_markets,
)

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Match: series prefix, YY, MMM, DD, HHMM, middle (teams + optional Gn),
# optional -suffix. See file docstring for the encoded format.
TICKER_RE = re.compile(
    r"^K[A-Z0-9]+-"
    r"(?P<yy>\d{2})(?P<mmm>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<time>\d{4})"
    r"(?P<middle>[A-Z]+(?:G\d+)?)"
    r"(?:-(?P<suffix>.+))?$"
)

GNUM_RE = re.compile(r"^(?P<core>[A-Z]+?)(?:G(?P<gnum>\d+))?$")


@dataclass(frozen=True)
class ParsedTicker:
    game_date: date
    time_hhmm: int
    home_code: str
    away_code: str
    game_number: int | None
    suffix: str | None


# --- Team-code catalogue ----------------------------------------------------


def load_team_codes() -> set[str]:
    """
    Return the set of 30 active MLB team abbreviations. Sourced from
    /teams (same source the schedule ingester uses) so the two agree
    on the exact set.

    Verified against the settled Kalshi corpus that these codes are 1:1
    with what Kalshi puts in tickers (AZ, ATH, KC, SF, CWS, ... all
    match).
    """
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/teams",
        params={"sportId": 1, "activeStatus": "Y"},
        timeout=10,
    )
    resp.raise_for_status()
    return {t["abbreviation"] for t in resp.json().get("teams", [])}


# --- Ticker parser ----------------------------------------------------------


def _split_teams(core: str, codes: set[str]) -> tuple[str, str] | None:
    """
    Split a two-team concatenation like 'MILPIT' or 'KCBAL' into
    (away, home). Deterministic because no MLB team code is a prefix of
    another (verified 30-code set).
    """
    for c in codes:
        if core.startswith(c):
            remainder = core[len(c):]
            if remainder in codes:
                return (c, remainder)
    return None


def parse_ticker(ticker: str, team_codes: set[str]) -> ParsedTicker | None:
    """
    Parse a Kalshi MLB market ticker. Returns None if the ticker doesn't
    match the expected shape or the middle can't be resolved against the
    team-code catalogue — the caller counts these as unparseable misses.
    """
    m = TICKER_RE.match(ticker)
    if not m:
        return None

    month = MONTHS.get(m.group("mmm"))
    if not month:
        return None

    year = 2000 + int(m.group("yy"))
    day = int(m.group("dd"))
    try:
        game_date = date(year, month, day)
    except ValueError:
        return None

    time_hhmm = int(m.group("time"))

    # Middle may end in 'G1'/'G2' — split off before team resolution.
    middle_m = GNUM_RE.match(m.group("middle"))
    if not middle_m:
        return None
    core = middle_m.group("core")
    gnum_str = middle_m.group("gnum")
    game_number = int(gnum_str) if gnum_str else None

    split = _split_teams(core, team_codes)
    if not split:
        return None
    away, home = split

    return ParsedTicker(
        game_date=game_date,
        time_hhmm=time_hhmm,
        home_code=home,
        away_code=away,
        game_number=game_number,
        suffix=m.group("suffix"),
    )


# --- Game lookup with doubleheader tie-break --------------------------------


def _find_candidates(
    conn,
    game_date: date,
    home_code: str,
    away_code: str,
) -> tuple[list[tuple[int, str | None, str | None]], str]:
    """
    Return (candidates, source) where candidates is a list of
    (game_pk, gameNumber, gameDate) tuples matching the date+teams key,
    and source describes which lookup path succeeded.

    Two lookups run in order:
        1. Direct match on `official_date` — the normal path.
        2. Match on `rescheduledFrom` date — Kalshi tickers encode the
           ORIGINALLY scheduled date, but a rain-out shifts MLB's
           `official_date` to when the game was actually played.
           `raw->>'rescheduledFrom'` on the games table carries the
           original date (as an ISO datetime); its DATE portion is what
           the Kalshi ticker matches.

    Empty result = the game was cancelled outright, or is outside our
    schedule-ingest window.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT game_pk,
                   raw->>'gameNumber' AS gnum,
                   raw->>'gameDate'   AS gdate
            FROM games
            WHERE official_date  = %s
              AND home_team_code = %s
              AND away_team_code = %s
            ORDER BY game_pk
            """,
            (game_date, home_code, away_code),
        )
        rows = list(cur.fetchall())
        if rows:
            return (rows, "official_date")

        # Fallback: this ticker's date may reference the ORIGINAL date of
        # a rescheduled game. `rescheduledFrom` is 'YYYY-MM-DDTHH:MM:SSZ';
        # substring the leading 10 chars to a DATE for the compare.
        cur.execute(
            """
            SELECT game_pk,
                   raw->>'gameNumber' AS gnum,
                   raw->>'gameDate'   AS gdate
            FROM games
            WHERE substring(raw->>'rescheduledFrom' FROM 1 FOR 10)::date = %s
              AND home_team_code = %s
              AND away_team_code = %s
            ORDER BY game_pk
            """,
            (game_date, home_code, away_code),
        )
        rows = list(cur.fetchall())
        return (rows, "rescheduled_from" if rows else "none")


def _pick_game(
    candidates: list[tuple[int, str | None, str | None]],
    parsed: ParsedTicker,
) -> tuple[int, str]:
    """
    Choose a game_pk from candidates. Returns (game_pk, tiebreak_reason)
    so we can audit which path resolved a doubleheader.

    Tie-break order:
        1. `game_number` from ticker matches gameNumber field.
        2. `time_hhmm` from ticker matches gameDate's UTC HHMM.
        3. Fallback: first by game_pk (arbitrary but deterministic).
    """
    if len(candidates) == 1:
        return (candidates[0][0], "single")

    # (1) Explicit game-number match
    if parsed.game_number is not None:
        for pk, gnum, _ in candidates:
            if gnum is not None and int(gnum) == parsed.game_number:
                return (pk, "gnum")

    # (2) Time match — gameDate is 'YYYY-MM-DDTHH:MM:SSZ'
    for pk, _, gdate in candidates:
        if gdate and len(gdate) >= 16:
            hhmm = int(gdate[11:13]) * 100 + int(gdate[14:16])
            if hhmm == parsed.time_hhmm:
                return (pk, "time")

    # (3) Deterministic fallback (least-game_pk from ORDER BY above)
    return (candidates[0][0], "fallback")


# --- Driver -----------------------------------------------------------------


def link_all(series_filter: str | None = None, dry_run: bool = False) -> None:
    team_codes = load_team_codes()

    with connect() as conn:
        unlinked = list_unlinked_markets(conn, series_filter)
        print(f"unlinked markets: {len(unlinked)}"
              + (f" (series={series_filter})" if series_filter else ""))

        n_matched = 0
        n_no_match = 0
        n_unparseable = 0
        n_no_game = 0
        tiebreak_counts: dict[str, int] = {}

        for ticker, _event in unlinked:
            parsed = parse_ticker(ticker, team_codes)
            if parsed is None:
                n_unparseable += 1
                continue

            candidates, source = _find_candidates(
                conn, parsed.game_date, parsed.home_code, parsed.away_code,
            )
            if not candidates:
                n_no_game += 1
                continue

            game_pk, reason = _pick_game(candidates, parsed)
            # Prepend the lookup source so we can see how many links
            # came via the rescheduled-from fallback in the summary.
            key = f"{source}/{reason}"
            tiebreak_counts[key] = tiebreak_counts.get(key, 0) + 1

            if not dry_run:
                insert_market_game_link(conn, ticker, game_pk, "ticker_parse")
            n_matched += 1

        if not dry_run:
            conn.commit()

    n_no_match = n_unparseable + n_no_game
    total = len(unlinked)
    pct = (100.0 * n_matched / total) if total else 0.0
    print()
    print(f"matched     : {n_matched:>7}  ({pct:.1f}%)")
    print(f"no match    : {n_no_match:>7}")
    print(f"  unparseable : {n_unparseable:>5}")
    print(f"  no game_pk  : {n_no_game:>5}")
    print()
    print("tie-break breakdown (matched only):")
    for reason, count in sorted(tiebreak_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<10}: {count}")
    if dry_run:
        print("\n(dry run — no rows written)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default=None,
                        help="Only link tickers in this series (e.g. KXMLBGAME)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report match rate without inserting rows")
    args = parser.parse_args()
    link_all(series_filter=args.series, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
