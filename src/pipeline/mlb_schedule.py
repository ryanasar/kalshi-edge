"""
src/pipeline/mlb_schedule.py

Ingester for MLB Stats API's /schedule endpoint, with the linescore /
probablePitcher / weather hydrations attached. One JSON call per date
covers everything we need on the outcome-labels side:

    • home_score, away_score, is_home_winner            (moneyline label)
    • innings[].{away.runs, home.runs}                   (F3/F5/F7/RFI labels)
    • total_runs, went_to_extras                         (totals / extras)
    • home/away probable pitcher IDs                     (feature key)
    • weather (temp, wind vector, condition)             (feature)

Endpoint:
    GET https://statsapi.mlb.com/api/v1/schedule
        ?sportId=1
        &date=YYYY-MM-DD
        &hydrate=linescore,probablePitcher,weather

No authentication required. Rate-limit is generous (not documented; treat
as unmetered, sleep 0.1s between dates as courtesy).

Usage:
    # single date
    ./.venv/bin/python -m src.pipeline.mlb_schedule --date 2026-07-11

    # date range (inclusive)
    ./.venv/bin/python -m src.pipeline.mlb_schedule \\
        --start 2026-05-01 --end 2026-07-12
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import requests

from src.pipeline.db import connect, upsert_game

BASE = "https://statsapi.mlb.com/api/v1"

# Courtesy pacing between date-scoped requests. MLB Stats API is
# unmetered in practice but the polite thing is to not hammer it.
DATE_SLEEP_S = 0.1

# Hydrations we need. If we ever add lineups/injuries this is the
# one place to touch — but per §4 first-pass scope, these three are it.
HYDRATIONS = "linescore,probablePitcher,weather"


# --- Team-code cache ---------------------------------------------------------
# MLB Stats API returns team IDs on every game; abbreviations live on the
# /teams endpoint. Fetch once at ingester startup, cache in-memory. The
# 30-team set is stable within a season (and effectively across seasons —
# rare rebrands, e.g., Cleveland Indians→Guardians).
_TEAM_ABBR: dict[int, str] = {}


def _load_team_abbreviations(season: int) -> dict[int, str]:
    """Return {team_id: abbreviation} for the given MLB season."""
    resp = requests.get(
        f"{BASE}/teams",
        params={"sportId": 1, "season": season, "activeStatus": "Y"},
        timeout=10,
    )
    resp.raise_for_status()
    return {t["id"]: t["abbreviation"] for t in resp.json().get("teams", [])}


def _team_code(team_id: int, official_date: date) -> str:
    """
    Team-id → Kalshi-compatible abbreviation. Verified against every
    KXMLBGAME ticker in the settled corpus that the two systems use
    identical codes (AZ, ATH, KC, SF, CWS, ...) — no translation needed.
    """
    if team_id not in _TEAM_ABBR:
        _TEAM_ABBR.update(_load_team_abbreviations(official_date.year))
    return _TEAM_ABBR[team_id]


# --- Outcome derivation ------------------------------------------------------


def _sum_runs(innings: list[dict], through: int) -> int | None:
    """
    Sum runs across both teams for the first `through` innings.
    Returns None if we don't have enough completed innings to compute
    (e.g., a game that ended after 4 innings can't answer F5).
    """
    if not innings or len(innings) < through:
        return None
    total = 0
    for inn in innings[:through]:
        # `runs` can legitimately be absent on the home half if the
        # home team didn't bat in the bottom of the final inning (walk-
        # off win) — treat missing as 0.
        total += (inn.get("away") or {}).get("runs") or 0
        total += (inn.get("home") or {}).get("runs") or 0
    return total


def _derive_outcomes(game: dict) -> dict:
    """
    Extract the outcome columns (home/away scores, is_home_winner, and
    the runs-through-N precomputes) from a hydrated /schedule game entry.

    Returns a dict of ONLY the outcome columns — the caller merges this
    onto the metadata dict. Every field is nullable; if the game isn't
    Final we return NULLs everywhere.
    """
    status = (game.get("status") or {}).get("abstractGameState", "")
    is_final = status == "Final"

    ls = game.get("linescore") or {}
    ls_teams = ls.get("teams") or {}
    innings = ls.get("innings") or []
    scheduled = ls.get("scheduledInnings") or game.get("scheduledInnings") or 9

    home_score = (ls_teams.get("home") or {}).get("runs")
    away_score = (ls_teams.get("away") or {}).get("runs")

    # Derive from scores rather than trusting MLB's `isWinner` field:
    # observed empirically that `linescore.teams.home.isWinner` is missing
    # on ~half of Final games in a sample day, while scores are always
    # present. Math > trusting a source of truth that's inconsistent.
    is_home_winner = None
    if is_final and home_score is not None and away_score is not None and home_score != away_score:
        is_home_winner = home_score > away_score

    # `went_to_extras` compares completed innings to scheduled, not to a
    # hardcoded 9 — 7-inning doubleheader games (post-2020 rule) can
    # still go to "extras" at 8+.
    went_to_extras = None
    if is_final:
        completed = len(innings)
        went_to_extras = completed > scheduled

    return {
        "home_score":      home_score if is_final else None,
        "away_score":      away_score if is_final else None,
        "is_home_winner":  is_home_winner if is_final else None,
        "total_runs":      (home_score + away_score) if (is_final and home_score is not None and away_score is not None) else None,
        "rfi_runs":        _sum_runs(innings, 1) if is_final else None,
        "f3_runs":         _sum_runs(innings, 3) if is_final else None,
        "f5_runs":         _sum_runs(innings, 5) if is_final else None,
        "f7_runs":         _sum_runs(innings, 7) if is_final else None,
        "went_to_extras":  went_to_extras,
        "innings":         innings if innings else None,
    }


def _game_to_row(game: dict) -> dict:
    """
    Transform a hydrated /schedule game entry into a dict matching the
    columns of the `games` table (kwargs for upsert_game).
    """
    official_date = date.fromisoformat(game["officialDate"])

    home = game["teams"]["home"]
    away = game["teams"]["away"]
    venue = game.get("venue") or {}

    # Probable pitchers are per-team on the hydrated schedule payload.
    home_pp = (home.get("probablePitcher") or {}).get("id")
    away_pp = (away.get("probablePitcher") or {}).get("id")

    row = {
        "game_pk":                  game["gamePk"],
        "official_date":            official_date,
        "game_type":                game.get("gameType", "R"),
        "status":                   (game.get("status") or {}).get("abstractGameState", ""),
        "scheduled_innings":        game.get("scheduledInnings") or 9,

        "home_team_id":             home["team"]["id"],
        "home_team_code":           _team_code(home["team"]["id"], official_date),
        "home_team_name":           home["team"]["name"],
        "away_team_id":             away["team"]["id"],
        "away_team_code":           _team_code(away["team"]["id"], official_date),
        "away_team_name":           away["team"]["name"],

        "venue_id":                 venue.get("id", 0),
        "venue_name":               venue.get("name", ""),

        "home_probable_pitcher_id": home_pp,
        "away_probable_pitcher_id": away_pp,

        "weather":                  game.get("weather"),

        "raw":                      game,
    }
    row.update(_derive_outcomes(game))
    return row


# --- Ingest driver -----------------------------------------------------------


def ingest_date(target: date, verbose: bool = True) -> tuple[int, int]:
    """
    Ingest every MLB game on `target`. Returns (n_fetched, n_upserted).
    """
    resp = requests.get(
        f"{BASE}/schedule",
        params={
            "sportId":  1,
            "date":     target.isoformat(),
            "hydrate":  HYDRATIONS,
        },
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()

    games: list[dict] = []
    for day in body.get("dates", []):
        games.extend(day.get("games", []))

    # Keep only regular-season games. The schedule endpoint also returns
    # the All-Star Game (gameType 'A'), spring training ('S'), and
    # postseason ('D'/'L'/'W'/'F'). The All-Star Game carries synthetic
    # "team" ids (159 = AL All-Stars, 160 = NL) that aren't in the 30-club
    # abbreviation map, so reaching _team_code with one raises KeyError and
    # kills the entire backfill mid-July every season. Filtering to 'R'
    # both fixes that and keeps the label universe aligned with the
    # Statcast feature window, which is already fetched with hfGT=R.
    reg_games = [g for g in games if g.get("gameType") == "R"]
    skipped = len(games) - len(reg_games)

    upserted = 0
    with connect() as conn:
        for g in reg_games:
            row = _game_to_row(g)
            if upsert_game(conn, row):
                upserted += 1
        conn.commit()

    if verbose:
        extra = f"  skipped_non_R={skipped}" if skipped else ""
        print(f"  {target.isoformat()}: fetched={len(games)}  "
              f"upserted={upserted}{extra}")
    return (len(games), upserted)


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Single date, YYYY-MM-DD")
    parser.add_argument("--start", help="Start date (inclusive)")
    parser.add_argument("--end",   help="End date (inclusive)")
    args = parser.parse_args()

    # Argument shape: either --date OR --start/--end. Enforce here so
    # confused invocations fail loudly at the boundary.
    if args.date and (args.start or args.end):
        parser.error("--date is exclusive with --start/--end")
    if not args.date and not (args.start and args.end):
        parser.error("Provide either --date, or --start AND --end")

    if args.date:
        start = end = date.fromisoformat(args.date)
    else:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)

    total_fetched = 0
    total_upserted = 0
    for d in _date_range(start, end):
        fetched, upserted = ingest_date(d)
        total_fetched  += fetched
        total_upserted += upserted
        time.sleep(DATE_SLEEP_S)

    print()
    print(f"Done. dates={((end - start).days) + 1}  "
          f"fetched={total_fetched}  upserted={total_upserted}")


if __name__ == "__main__":
    main()
