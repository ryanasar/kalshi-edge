"""
src/pipeline/backfill_candles.py

Walks every ticker in the `markets` table and pulls its candles into
`market_candles`. Idempotent by DB constraint (safe to re-run).

Default behavior: pull each market's FULL life (open_time → close_time).
Pass `--hours-before-close N` to shorten to the last N hours before close
(the calibration-minimum window).

Usage:
    # archive-complete: every candle for every market's entire life
    ./.venv/bin/python -m src.pipeline.backfill_candles

    # calibration-minimum: just the last 24h before close
    ./.venv/bin/python -m src.pipeline.backfill_candles --hours-before-close 24

    # only KXFED tickers (any window)
    ./.venv/bin/python -m src.pipeline.backfill_candles --series KXFED
"""

from __future__ import annotations

import argparse
import time
import traceback

from src.pipeline.db import connect
from src.pipeline.ingest_candles import ingest_ticker

# Kalshi ~= 20 req/sec. One candles request per ticker; 0.15s sleep = ~6.5
# req/sec, safe even during bursts.
TICKER_SLEEP_S = 0.15


def list_tickers(series_filter: str | None) -> list[str]:
    with connect() as conn, conn.cursor() as cur:
        if series_filter:
            cur.execute(
                "SELECT ticker FROM markets WHERE series_ticker = %s ORDER BY ticker",
                (series_filter,),
            )
        else:
            cur.execute("SELECT ticker FROM markets ORDER BY ticker")
        return [row[0] for row in cur.fetchall()]


def backfill(
    period_minutes: int,
    hours_before_close: int | None,
    series_filter: str | None,
) -> None:
    tickers = list_tickers(series_filter)
    window_desc = (
        "full life (open→close)"
        if hours_before_close is None
        else f"{hours_before_close}h before close"
    )
    print(
        f"Backfilling candles for {len(tickers)} tickers  "
        f"(period={period_minutes}min, window={window_desc})"
    )

    ok = 0
    err = 0
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        try:
            ingest_ticker(ticker, period_minutes, hours_before_close)
            ok += 1
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            err += 1
        time.sleep(TICKER_SLEEP_S)

    print(f"\nDone. {ok} ok, {err} errored.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period", type=int, default=60,
        help="Candle length in minutes (default 60 = hourly)",
    )
    parser.add_argument(
        "--hours-before-close", type=int, default=None,
        help=(
            "If set, only pull the last N hours before each market's "
            "close_time. Default (unset) = pull each market's entire life "
            "from open_time to close_time — this is the archive-complete mode."
        ),
    )
    parser.add_argument(
        "--series", default=None,
        help="Only backfill tickers in this series (e.g. KXFED)",
    )
    args = parser.parse_args()
    backfill(args.period, args.hours_before_close, args.series)


if __name__ == "__main__":
    main()
