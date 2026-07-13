"""
src/pipeline/backfill_series.py

Iterates over candidate Tier 1 series tickers (per CLAUDE.md §3) and calls
`ingest_series` on each. Tolerates unknown-series and empty-series responses
as normal outcomes — the whole point is to discover which of our guesses at
Kalshi's series-ticker convention actually resolve to real markets.

Usage:
    ./.venv/bin/python -m src.pipeline.backfill_series
    ./.venv/bin/python -m src.pipeline.backfill_series KXCPI KXPCE   # subset

Discovery is iterative: any ticker in the CANDIDATE_SERIES list that returns
0 markets is either not the right name or genuinely has no settled markets
yet. Both cases we skip and log; a human iterates on the list.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from src.pipeline.ingest_markets import ingest_series

# MLB market series on Kalshi. Confirmed via /events archival crawl.
# Grouped by market type. See CLAUDE.md §3 for scope discussion.
CANDIDATE_SERIES: list[str] = [
    # ----- game outcomes (modeling focus) -----
    "KXMLBGAME",      # Moneyline — who wins
    "KXMLBTOTAL",     # Full-game total runs (over/under)
    "KXMLBSPREAD",    # Run line
    "KXMLBTEAMTOTAL", # Per-team total runs
    # ----- N-innings sub-markets (starter-pitcher dominated) -----
    "KXMLBF3",        # First 3 innings totals
    "KXMLBF5",        # First 5 innings moneyline
    "KXMLBF5SPREAD",  # First 5 innings run line
    "KXMLBF5TOTAL",   # First 5 innings totals
    "KXMLBF7",        # First 7 innings
    "KXMLBRFI",       # Runs in first inning
    "KXMLBEXTRAS",    # Extras (game goes to extra innings)
    # ----- player props (ingest for archive; deferred for modeling) -----
    "KXMLBHR",        # Home runs (player prop)
    "KXMLBHIT",       # Hits (player prop)
    "KXMLBHRR",       # HR + Runs (player prop)
    "KXMLBKS",        # Strikeouts (pitcher prop)
    "KXMLBOUTS",      # Outs recorded (pitcher prop)
    "KXMLBRBI",       # RBIs (player prop)
    "KXMLBSB",        # Stolen bases (player prop)
    "KXMLBTB",        # Total bases (player prop)
]

# One second between series so we're comfortably under Kalshi's 20 req/sec
# even with a 200-market page burst inside each series.
SERIES_SLEEP_S = 1.0


def backfill(series_list: list[str]) -> None:
    results: list[tuple[str, str]] = []
    for series in series_list:
        print(f"\n=== {series} ===")
        try:
            ingest_series(series)
            results.append((series, "ok"))
        except SystemExit:
            raise
        except Exception:
            # Print the full trace so we can see what went wrong without
            # letting one bad series abort the whole backfill.
            traceback.print_exc()
            results.append((series, "error"))
        time.sleep(SERIES_SLEEP_S)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for series, status in results:
        print(f"  {series:<20} {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "series",
        nargs="*",
        help=f"Optional subset. Default = all {len(CANDIDATE_SERIES)} candidates.",
    )
    args = parser.parse_args()
    series_list = args.series or CANDIDATE_SERIES
    backfill(series_list)
    sys.exit(0)


if __name__ == "__main__":
    main()
