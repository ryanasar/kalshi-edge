"""
src/trading/book_watcher.py

The live market-data layer, assembled. It wires the WS frame stream
(ws_client.stream_session) into the OrderBook state machine and a tick
writer, giving:

  1. a live top-of-book view that updates in real time, and
  2. lossless tick persistence into `market_ticks` — the forward-only
     dataset market-making simulation needs.

Reliability: the reconnect loop owns both failure modes. A dropped
connection reconnects; a SEQUENCE GAP (OrderBook.apply → False) tears down
the session and resubscribes, because after a gap the only way back to a
correct book is a fresh snapshot. Buffered ticks are flushed on every exit
path so a crash loses at most one buffer.

DB writes run via asyncio.to_thread so the synchronous psycopg insert never
blocks the event loop that's consuming the socket.

Usage:
    # auto-discover an open market, watch + persist
    ./.venv/bin/python -m src.trading.book_watcher --seconds 20

    # a specific market, no persistence (just watch)
    ./.venv/bin/python -m src.trading.book_watcher --ticker KXMLB... --no-persist
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone

from psycopg.types.json import Json

from src.pipeline.auth import KalshiClient
from src.pipeline.db import connect
from src.trading.order_book import OrderBook
from src.trading.ws_client import (
    REST_HOST,
    _discover_active_markets,
    stream_session,
)

BOOK_TYPES = {"orderbook_snapshot", "orderbook_delta"}
CHANNELS = ("orderbook_delta", "ticker", "trade")


class TickWriter:
    """Buffers ticks and flushes them to `market_ticks` in batches, off the
    event loop. Extracts the common query columns; keeps the full frame in
    `raw` (JSONB) so nothing is lost."""

    def __init__(self, flush_every: int = 10):
        self.buf: list[tuple] = []
        self.flush_every = flush_every

    def add(self, ticker: str, frame: dict, recv_ts: datetime) -> bool:
        msg = frame.get("msg", {})
        # size: delta for book deltas, count for trades, else None.
        size = msg.get("delta_fp", msg.get("count"))
        self.buf.append((
            ticker, recv_ts, msg.get("ts_ms"), frame.get("seq"),
            frame.get("type"), msg.get("side"),
            msg.get("price_dollars"), size, Json(frame),
        ))
        return len(self.buf) >= self.flush_every

    async def flush(self) -> int:
        if not self.buf:
            return 0
        batch, self.buf = self.buf, []
        # shield: if the watcher is being cancelled by the timeout safety
        # net, still let the DB write finish rather than lose the batch.
        await asyncio.shield(asyncio.to_thread(self._write, batch))
        return len(batch)

    @staticmethod
    def _write(batch: list[tuple]) -> None:
        with connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO market_ticks
                   (ticker, recv_ts, exch_ts_ms, seq, type, side, price, size, raw)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                batch,
            )
            conn.commit()


def _print_top(book: OrderBook) -> None:
    t = book.top()
    bid = f"{t['yes_bid']:.4f}" if t["yes_bid"] is not None else "  —   "
    ask = f"{t['yes_ask']:.4f}" if t["yes_ask"] is not None else "  —   "
    spr = f"{t['spread']:.4f}" if t["spread"] is not None else "  —  "
    print(f"  {t['ticker'][-24:]:>24}  bid {bid}  ask {ask}  spread {spr}")


async def watch(tickers: list[str], persist: bool = True,
                max_seconds: int = 0) -> None:
    client = KalshiClient.from_env(REST_HOST)
    # Recorder mode: seq is a global per-connection counter across all these
    # markets, so per-market gap detection is off (see OrderBook.check_seq).
    books: dict[str, OrderBook] = {t: OrderBook(t, check_seq=False) for t in tickers}
    writer = TickWriter() if persist else None
    total = 0
    start = time.monotonic()
    print(f"watching {len(tickers)} market(s)  (persist={persist})")

    while True:  # reconnect / resubscribe loop
        try:
            async for frame in stream_session(client, tickers, CHANNELS):
                if writer and writer.add(frame.get("msg", {}).get("market_ticker", ""),
                                         frame, datetime.now(timezone.utc)):
                    total += await writer.flush()
                if frame.get("type") in BOOK_TYPES:
                    book = books.get(frame["msg"]["market_ticker"])
                    if book is not None:
                        book.apply(frame)  # recorder mode: always applies
                        # Only surface markets with a real two-sided quote.
                        if book.mid() is not None:
                            _print_top(book)
                if max_seconds and time.monotonic() - start >= max_seconds:
                    if writer:
                        total += await writer.flush()
                    print(f"\nstopped after {max_seconds}s. persisted {total} ticks.")
                    return
        except (OSError, asyncio.IncompleteReadError) as e:
            print(f"  connection error ({type(e).__name__}); reconnecting")
        except Exception as e:  # noqa: BLE001 — surface, flush, then retry
            print(f"  session error ({type(e).__name__}: {e}); reconnecting")
        finally:
            if writer:
                total += await writer.flush()
        await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="Single market; auto-discovers a batch if omitted")
    parser.add_argument("--n", type=int, default=100,
                        help="How many open markets to subscribe to when auto-discovering")
    parser.add_argument("--seconds", type=int, default=0,
                        help="Stop after N seconds (0 = run until interrupted)")
    parser.add_argument("--no-persist", action="store_true",
                        help="Watch only; do not write to market_ticks")
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = _discover_active_markets(KalshiClient.from_env(REST_HOST), args.n)
        if not tickers:
            raise SystemExit("No open markets found to watch right now.")

    async def _bounded() -> None:
        coro = watch(tickers, persist=not args.no_persist, max_seconds=args.seconds)
        # Safety net: if the market is dead silent, recv() blocks and the
        # internal deadline never fires — wait_for guarantees we still stop.
        if args.seconds > 0:
            await asyncio.wait_for(coro, timeout=args.seconds + 8)
        else:
            await coro

    try:
        asyncio.run(_bounded())
    except (KeyboardInterrupt, asyncio.TimeoutError):
        print("\nstopped (timeout safety net — market was quiet).")


if __name__ == "__main__":
    main()
