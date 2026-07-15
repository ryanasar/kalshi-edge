"""
src/pipeline/backfill_candles.py

Walks every ticker in the `markets` table and pulls its candles into
`market_candles`. Idempotent by DB constraint (safe to re-run).

Default behavior: pull each market's FULL life (open_time → close_time).
Pass `--hours-before-close N` to shorten to the last N hours before close
(the calibration-minimum window).

Usage:
    # archive-complete, single-threaded (legacy, ~5h wall-clock for 12.5k tickers)
    ./.venv/bin/python -m src.pipeline.backfill_candles

    # fast path: 8 worker threads sharing a 15 req/s token bucket
    ./.venv/bin/python -m src.pipeline.backfill_candles --workers 8

    # calibration-minimum: just the last 24h before close, parallelized
    ./.venv/bin/python -m src.pipeline.backfill_candles --workers 8 --hours-before-close 24

    # only KXFED tickers
    ./.venv/bin/python -m src.pipeline.backfill_candles --series KXFED

Parallelism model:
    - Threads, not asyncio: KalshiClient is sync `requests` and the workload
      is I/O bound. A ThreadPoolExecutor gets the same speedup as aiohttp
      without rewriting auth.
    - Token-bucket limiter shared across workers, capped below Kalshi's
      ~20 req/s ceiling — see RATE_LIMIT_HZ. Going above 20 gets 429s.
    - Per-worker KalshiClient (own requests.Session) and per-worker psycopg
      connection — neither is thread-safe when shared.
    - Fetch + insert stay coupled per ticker; DB is not the bottleneck at
      this scale, so a producer/consumer split would just add complexity.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.pipeline.db import connect, get_market_bounds, insert_candles

PROD_HOST = CANDIDATE_HOSTS[0]

# Kalshi's documented rate ceiling is ~20 req/s on the basic tier. Aim for
# 15 to leave headroom for jitter and for the other pipeline processes
# (schedule ingester, WebSocket, ad-hoc scripts) that share the credential.
RATE_LIMIT_HZ = 15.0


class TokenBucket:
    """
    Classic token-bucket limiter. `rate_hz` tokens refilled per second, up
    to `capacity` in reserve. `acquire()` blocks until one token is available.

    Sharing one instance across threads is the point — each worker calls
    acquire() before issuing an HTTP request, and the bucket serializes
    them to a fixed aggregate request rate no matter how many threads are
    contending.
    """
    def __init__(self, rate_hz: float, capacity: float | None = None):
        self.rate_hz = rate_hz
        self.capacity = capacity if capacity is not None else rate_hz
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_hz)
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                # Time until a full token is available. Sleep OUTSIDE the
                # lock so other workers can keep contending.
                wait = (1.0 - self.tokens) / self.rate_hz
            time.sleep(wait)


# Each worker thread gets its own KalshiClient (own requests.Session) and
# its own Postgres connection. Held in thread-local storage so we build
# them lazily on first use per worker and reuse across tickers.
_worker_local = threading.local()


def _worker_client() -> KalshiClient:
    client = getattr(_worker_local, "client", None)
    if client is None:
        client = KalshiClient.from_env(host=PROD_HOST)
        _worker_local.client = client
    return client


def _worker_conn():
    conn = getattr(_worker_local, "conn", None)
    if conn is None:
        conn = connect()
        _worker_local.conn = conn
    return conn


def _fetch_and_insert(
    ticker: str,
    period_minutes: int,
    hours_before_close: int | None,
    bucket: TokenBucket,
) -> tuple[str, int, int]:
    """
    Worker-side unit: fetch one ticker's candles and insert them. Returns
    (ticker, fetched_count, inserted_count). Raises on failure — the caller
    catches and counts.

    Duplicates the shape of `ingest_candles.ingest_ticker` because that
    function opens its own connection per call (sensible for the serial
    CLI, wasteful when we already have a per-worker connection).
    """
    client = _worker_client()
    conn = _worker_conn()

    series_ticker = ticker.split("-", 1)[0]

    bounds = get_market_bounds(conn, ticker)
    if bounds is None:
        raise RuntimeError(
            f"Ticker {ticker!r} is not in the markets table. "
            f"Run `ingest_markets {series_ticker}` first."
        )
    open_time, close_time = bounds

    window_end = close_time
    window_start = (
        open_time
        if hours_before_close is None
        else window_end - timedelta(hours=hours_before_close)
    )

    params = {
        "start_ts": int(window_start.timestamp()),
        "end_ts": int(window_end.timestamp()),
        "period_interval": period_minutes,
    }
    path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"

    # One token per HTTP request. Blocks the worker if the bucket is empty,
    # which is the intended backpressure.
    bucket.acquire()
    resp = client.request("GET", path, params=params)
    resp.raise_for_status()
    candles = resp.json().get("candlesticks", [])

    inserted = insert_candles(conn, ticker, period_minutes, candles)
    conn.commit()
    return ticker, len(candles), inserted


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
    workers: int,
) -> None:
    tickers = list_tickers(series_filter)
    window_desc = (
        "full life (open→close)"
        if hours_before_close is None
        else f"{hours_before_close}h before close"
    )
    print(
        f"Backfilling candles for {len(tickers)} tickers  "
        f"(period={period_minutes}min, window={window_desc}, "
        f"workers={workers}, rate_cap={RATE_LIMIT_HZ:.0f} req/s)"
    )

    bucket = TokenBucket(rate_hz=RATE_LIMIT_HZ)
    ok = 0
    err = 0
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_and_insert,
                ticker,
                period_minutes,
                hours_before_close,
                bucket,
            ): ticker
            for ticker in tickers
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            ticker = futures[fut]
            try:
                _, fetched, inserted = fut.result()
                ok += 1
                print(
                    f"[{i}/{len(tickers)}] {ticker}: "
                    f"fetched {fetched}, inserted {inserted}"
                )
            except Exception:
                err += 1
                print(f"[{i}/{len(tickers)}] {ticker}: FAILED")
                traceback.print_exc()

    elapsed = time.monotonic() - started
    print(
        f"\nDone. {ok} ok, {err} errored, wall-clock {elapsed:.1f}s "
        f"({ok / elapsed:.2f} ticker/s)."
    )


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
    parser.add_argument(
        "--workers", type=int, default=8,
        help=(
            "Number of concurrent worker threads. Default 8. "
            "Set to 1 for the legacy single-threaded behavior. "
            f"Aggregate request rate is capped at {RATE_LIMIT_HZ:.0f} req/s "
            "by a shared token bucket regardless of worker count."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    backfill(args.period, args.hours_before_close, args.series, args.workers)


if __name__ == "__main__":
    main()
