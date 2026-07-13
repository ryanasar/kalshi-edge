"""
src/pipeline/backfill_all.py

Full-Kalshi archival crawler. Discovers every series that has ever
settled via paginated /events, then ingests all settled markets for each.

Purpose is preservation, not modeling. Kalshi's retention window is ~2
months; every day this doesn't run, some history quietly disappears
from their public endpoints. Running this daily builds a proprietary
archive of resolved markets that will outlive Kalshi's own retention.

Intentionally separate from `backfill_series` so the Tier 1 modeling
workflow (CANDIDATE_SERIES = macro releases we actively model) stays
clean and small. This script is the "just in case" flywheel.

Usage:
    ./.venv/bin/python -m src.pipeline.backfill_all
    ./.venv/bin/python -m src.pipeline.backfill_all --max-event-pages 50

Recommended cadence: daily cron (macOS launchd or a Python-side
scheduler). Runtime ~30–60 min for a full sweep; individual runs after
day 1 are much faster because most markets are already in the DB and
`ON CONFLICT DO NOTHING` short-circuits the inserts.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.ingest_markets import ingest_series

PROD_HOST = CANDIDATE_HOSTS[0]

# Sleeps calibrated to stay well under Kalshi's ~20 req/sec even during
# bursts inside each series ingestion.
EVENT_PAGE_SLEEP_S = 0.2
SERIES_SLEEP_S = 1.0


def discover_series(max_event_pages: int) -> list[tuple[str, int]]:
    """
    Paginate /events?status=settled and return (series_ticker, event_count)
    sorted by event_count descending — heaviest series first so if we bail
    early we've captured the most.
    """
    client = KalshiClient.from_env(host=PROD_HOST)
    counts: Counter = Counter()
    cursor = ""
    for page in range(1, max_event_pages + 1):
        params: dict = {"limit": 200, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        resp = client.request("GET", "/events", params=params)
        resp.raise_for_status()
        body = resp.json()

        for e in body.get("events", []):
            s = e.get("series_ticker")
            if s:
                counts[s] += 1

        cursor = body.get("cursor") or ""
        print(
            f"  discovery page {page:>2}: "
            f"cumulative events={sum(counts.values())}  "
            f"distinct series={len(counts)}"
        )
        if not cursor:
            break
        time.sleep(EVENT_PAGE_SLEEP_S)

    return counts.most_common()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-event-pages",
        type=int,
        default=30,
        help="Cap on discovery pagination (default 30 = up to 6000 events)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover series and print the plan, but skip the ingest.",
    )
    args = parser.parse_args()

    print("=== discovery ===")
    ranked = discover_series(args.max_event_pages)
    print(f"\nDiscovered {len(ranked)} distinct series across settled events.")

    if args.dry_run:
        for s, n in ranked:
            print(f"  {n:>5}  {s}")
        return

    print("\n=== ingest ===")
    for i, (series, event_count) in enumerate(ranked, start=1):
        header = f"[{i}/{len(ranked)}] {series}  (events≈{event_count})"
        print(f"\n{'-' * len(header)}\n{header}\n{'-' * len(header)}")
        try:
            ingest_series(series)
        except SystemExit:
            raise
        except Exception as exc:
            # Log and continue — one bad series doesn't kill the archive.
            print(f"  ERROR on {series}: {exc}")
        time.sleep(SERIES_SLEEP_S)

    print("\n=== done ===")


if __name__ == "__main__":
    main()
