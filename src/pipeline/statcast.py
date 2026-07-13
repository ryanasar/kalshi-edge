"""
src/pipeline/statcast.py

Ingester for Baseball Savant pitch-level Statcast data. Public CSV
endpoint, no auth, no documented rate limit — we sleep 0.5s between days
as courtesy.

Endpoint (URL-encoded arg soup — the important bits are the date bounds
and game_type):

    GET https://baseballsavant.mlb.com/statcast_search/csv
        ?all=true                       # include every column
        &hfGT=R|                        # game type: 'R' = regular season
        &hfSea={year}|                  # season filter (Savant requires it)
        &game_date_gt=YYYY-MM-DD        # inclusive lower bound
        &game_date_lt=YYYY-MM-DD        # inclusive upper bound
        &type=details                   # per-pitch (not per-summary) output
        &player_type=pitcher            # required; doesn't filter rows
        &min_pitches=0
        &min_results=0
        &min_pas=0
        &sort_col=pitches
        &sort_order=desc
        &group_by=name

CSV has 117 columns; we keep the ~30 in db.STATCAST_COLUMNS (the physical
primitives, expected stats, and identifiers needed for pitcher / batter
aggregation). Other columns (fielders, umpire, delta win-exp, alignments)
are second-pass — the raw CSV is not persisted anywhere.

Usage:
    # single date
    ./.venv/bin/python -m src.pipeline.statcast --date 2025-07-11

    # date range (inclusive)
    ./.venv/bin/python -m src.pipeline.statcast --start 2024-03-28 --end 2024-10-01

    # default season backfill (2024, 2025, 2026-YTD)
    ./.venv/bin/python -m src.pipeline.statcast --default-window
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from datetime import date, timedelta

import requests

from src.pipeline.db import STATCAST_COLUMNS, connect, insert_pitches

CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Courtesy pacing between per-date requests. Empirically Savant handles
# ~2 req/sec without pushback; 0.5s is a safe default.
DATE_SLEEP_S = 0.5

# Which columns need numeric-int coercion vs. string vs. numeric-decimal.
# We split them out explicitly so a single malformed row doesn't get
# silently downcast to float. Columns not listed here pass through as
# strings (psycopg auto-coerces to NUMERIC / TEXT as needed).
INT_COLUMNS = {
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "inning",
    "outs_when_up",
    "balls",
    "strikes",
    "zone",
    "hit_distance_sc",
}


def _empty_to_none(v: str) -> str | None:
    """Savant emits '' for missing values; DB wants NULL. One-liner isolate
    so we don't accidentally treat literal '0' or '0.0' as missing."""
    return None if v == "" else v


def _coerce(row: dict[str, str]) -> tuple:
    """
    Extract STATCAST_COLUMNS from a CSV row (in order), converting empty
    strings to NULL and casting integer columns to int. Numeric-decimal
    columns pass through as strings — psycopg will coerce to NUMERIC.
    """
    out: list = []
    for col in STATCAST_COLUMNS:
        v = _empty_to_none(row.get(col, ""))
        if v is None:
            out.append(None)
        elif col in INT_COLUMNS:
            # Some Savant integer columns are emitted as '32.0' (float-
            # style). int(float(v)) is a small ceremony that survives both.
            out.append(int(float(v)))
        elif col == "game_date":
            out.append(date.fromisoformat(v))
        else:
            out.append(v)
    return tuple(out)


def _fetch_csv(target: date) -> list[dict[str, str]]:
    """
    Fetch one day of Statcast pitches. Returns list of dict rows (CSV
    header = keys). Empty list on 0-row days (offseason, all-star break).

    Uses `--data-urlencode`-equivalent through requests' `params` dict
    so special characters (the required `|` on hfGT / hfSea) survive.
    """
    resp = requests.get(
        CSV_URL,
        params={
            "all":            "true",
            "hfGT":           "R|",
            "hfSea":          f"{target.year}|",
            "game_date_gt":   target.isoformat(),
            "game_date_lt":   target.isoformat(),
            "type":           "details",
            "player_type":    "pitcher",
            "min_pitches":    "0",
            "min_results":    "0",
            "min_pas":        "0",
            "sort_col":       "pitches",
            "sort_order":     "desc",
            "group_by":       "name",
        },
        timeout=90,
    )
    resp.raise_for_status()
    text = resp.text

    # Savant sometimes emits a BOM (U+FEFF) as the first char of the CSV.
    # csv.DictReader treats it as part of the first column name, so
    # `row["pitch_type"]` misses. Strip it once, up front.
    if text.startswith("﻿"):
        text = text.lstrip("﻿")

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def ingest_date(target: date, verbose: bool = True) -> tuple[int, int]:
    """
    Ingest one date of Statcast pitches. Returns (n_fetched, n_inserted).
    """
    rows_dict = _fetch_csv(target)
    if not rows_dict:
        if verbose:
            print(f"  {target.isoformat()}: 0 pitches (no games / no data)",
                  flush=True)
        return (0, 0)

    tuples = [_coerce(r) for r in rows_dict]

    with connect() as conn:
        inserted = insert_pitches(conn, tuples)
        conn.commit()

    if verbose:
        print(
            f"  {target.isoformat()}: fetched={len(tuples):>5}  "
            f"inserted={inserted:>5}  "
            f"(conflicts={len(tuples) - inserted})",
            flush=True,
        )
    return (len(tuples), inserted)


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# --- Season windows ----------------------------------------------------------
# 2023 rule changes (pitch clock, larger bases, shift restrictions) shifted
# the run environment materially; pre-2023 data belongs in a follow-up
# training window with its own error distribution. First-pass backfill is
# the post-rule-change era plus 2026 YTD.
DEFAULT_WINDOWS = [
    (date(2024, 3, 28), date(2024, 10, 1)),   # 2024 regular season
    (date(2025, 3, 27), date(2025, 10, 1)),   # 2025 regular season
    (date(2026, 3, 26), date.today()),        # 2026 YTD
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Single date, YYYY-MM-DD")
    parser.add_argument("--start", help="Start date (inclusive)")
    parser.add_argument("--end",   help="End date (inclusive)")
    parser.add_argument(
        "--default-window",
        action="store_true",
        help="Run the built-in 2024 + 2025 + 2026-YTD backfill",
    )
    args = parser.parse_args()

    if args.default_window:
        windows = DEFAULT_WINDOWS
    else:
        if args.date and (args.start or args.end):
            parser.error("--date is exclusive with --start/--end")
        if not args.date and not (args.start and args.end):
            parser.error("Provide --default-window, --date, or --start AND --end")
        if args.date:
            s = e = date.fromisoformat(args.date)
        else:
            s = date.fromisoformat(args.start)
            e = date.fromisoformat(args.end)
        windows = [(s, e)]

    total_fetched = 0
    total_inserted = 0
    errored: list[tuple[date, str]] = []
    for (s, e) in windows:
        print(f"\n=== {s.isoformat()} → {e.isoformat()} ===", flush=True)
        for d in _date_range(s, e):
            # One transient Savant 5xx or network blip shouldn't kill the
            # whole backfill. Log and continue; the caller can re-run for
            # missed dates (idempotent via ON CONFLICT).
            try:
                fetched, inserted = ingest_date(d)
                total_fetched  += fetched
                total_inserted += inserted
            except Exception as ex:  # noqa: BLE001
                print(f"  {d.isoformat()}: ERROR {type(ex).__name__}: {ex}",
                      flush=True)
                errored.append((d, f"{type(ex).__name__}: {ex}"))
            time.sleep(DATE_SLEEP_S)

    if errored:
        print(f"\n{len(errored)} dates errored (re-runnable):", flush=True)
        for d, msg in errored:
            print(f"  {d.isoformat()}: {msg}", flush=True)

    print()
    print(f"Done. fetched={total_fetched}  inserted={total_inserted}  "
          f"(existing skipped: {total_fetched - total_inserted})")


if __name__ == "__main__":
    main()
