"""
src/pipeline/ingest_candles.py

Fetch historical candles for a single market ticker and write them into
`market_candles`. Idempotent.

Default behavior: pull the market's ENTIRE life (from `open_time` to
`close_time` in the DB). Pass `--hours-before-close N` to shorten to just
the last N hours before close.

Usage:
    # full life (archive-complete)
    ./.venv/bin/python -m src.pipeline.ingest_candles KXFED-26JUN-T4.25

    # last 24h only (calibration-minimum)
    ./.venv/bin/python -m src.pipeline.ingest_candles KXFED-26JUN-T4.25 \
        --hours-before-close 24

Notes:
    - Requires that the ticker already exists in `markets` (FK constraint).
      Run `ingest_markets` for the series first.
    - Kalshi's candlesticks endpoint skips periods with no activity — the
      returned candle count can be much less than the number of hours in
      the window.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.db import connect, get_market_bounds, insert_candles

PROD_HOST = CANDIDATE_HOSTS[0]


def ingest_ticker(
    ticker: str,
    period_minutes: int = 60,
    hours_before_close: int | None = None,
) -> None:
    """
    Fetch candles for one ticker and upsert into `market_candles`.

    `hours_before_close=None` (default) means "full life": window runs
    from the market's `open_time` to its `close_time`. Passing a positive
    integer narrows the window to the last N hours before close.
    """
    client = KalshiClient.from_env(host=PROD_HOST)

    # Kalshi's endpoint is /series/{s}/markets/{t}/candlesticks — series
    # comes from ticker prefix, same rule as ingest_markets.
    series_ticker = ticker.split("-", 1)[0]

    with connect() as conn:
        bounds = get_market_bounds(conn, ticker)
        if bounds is None:
            raise SystemExit(
                f"Ticker {ticker!r} is not in the markets table. "
                f"Run `ingest_markets {series_ticker}` first."
            )
        open_time, close_time = bounds

        # Default: pull the market's ENTIRE life. This is what an archival
        # backfill wants — we lose Kalshi's history the moment they purge.
        # A narrower window (--hours-before-close) is available for the
        # calibration-only "give me the close point" case where we don't
        # need intraday history.
        window_end = close_time
        if hours_before_close is None:
            window_start = open_time
            window_desc = "full life (open→close)"
        else:
            window_start = window_end - timedelta(hours=hours_before_close)
            window_desc = f"{hours_before_close}h before close"

        params = {
            "start_ts": int(window_start.timestamp()),
            "end_ts": int(window_end.timestamp()),
            "period_interval": period_minutes,
        }
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"

        resp = client.request("GET", path, params=params)
        resp.raise_for_status()
        body = resp.json()
        candles = body.get("candlesticks", [])

        inserted = insert_candles(conn, ticker, period_minutes, candles)
        conn.commit()

    print(
        f"{ticker}: fetched {len(candles)} candles "
        f"(period={period_minutes}min, window={window_desc}), "
        f"inserted {inserted} new rows."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="e.g. KXFED-26JUN-T4.25")
    parser.add_argument(
        "--period",
        type=int,
        default=60,
        help="Candle length in minutes (default 60 = hourly)",
    )
    parser.add_argument(
        "--hours-before-close",
        type=int,
        default=None,
        help=(
            "If set, only pull the last N hours before close. "
            "Default (unset) = pull the market's entire life from open→close."
        ),
    )
    args = parser.parse_args()
    ingest_ticker(args.ticker, args.period, args.hours_before_close)


if __name__ == "__main__":
    main()
