"""
src/pipeline/ingest_markets.py

Batch-ingest all settled markets for a series from Kalshi into the `markets`
table. Idempotent (ON CONFLICT DO NOTHING).

Usage:
    ./.venv/bin/python -m src.pipeline.ingest_markets KXFED
    ./.venv/bin/python -m src.pipeline.ingest_markets KXCPI --page-size 100
"""

from __future__ import annotations

import argparse
import time

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.db import connect, insert_market

# First entry is the confirmed prod host; the list remains from the auth-
# discovery phase. When auth.py is cleaned up, this becomes `PROD_HOST`.
PROD_HOST = CANDIDATE_HOSTS[0]

# Kalshi basic tier ~= 20 reads/sec. Sleeping 0.1s between pages caps us at
# ~10 pages/sec — comfortably under the limit even before any 429 handling.
PAGE_SLEEP_S = 0.1


def ingest_series(series_ticker: str, page_size: int = 200) -> None:
    client = KalshiClient.from_env(host=PROD_HOST)

    fetched = 0
    inserted = 0
    cursor = ""
    page_num = 0

    with connect() as conn:
        while True:
            page_num += 1
            params: dict = {
                "series_ticker": series_ticker,
                "status": "settled",
                "limit": page_size,
            }
            # First page: no cursor. Subsequent pages: whatever Kalshi returned.
            if cursor:
                params["cursor"] = cursor

            resp = client.request("GET", "/markets", params=params)
            resp.raise_for_status()
            body = resp.json()

            page = body.get("markets", [])
            new_this_page = 0
            for m in page:
                fetched += 1
                if insert_market(conn, m):
                    inserted += 1
                    new_this_page += 1

            # Commit every page so a mid-run failure doesn't cost us
            # everything ingested so far.
            conn.commit()

            print(
                f"page {page_num:>3}: fetched={len(page):>4}  "
                f"new={new_this_page:>4}  "
                f"(running total: fetched={fetched}, new={inserted})"
            )

            cursor = body.get("cursor") or ""
            if not cursor:
                break
            time.sleep(PAGE_SLEEP_S)

    print(
        f"\nDone. series_ticker={series_ticker}  "
        f"fetched={fetched} markets, inserted {inserted} new rows "
        f"({fetched - inserted} already existed)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series_ticker", help="e.g. KXFED, KXCPI, KXPCE")
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Markets per Kalshi page (default 200; max varies by endpoint)",
    )
    args = parser.parse_args()
    ingest_series(args.series_ticker, args.page_size)


if __name__ == "__main__":
    main()
