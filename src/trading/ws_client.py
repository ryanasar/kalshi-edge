"""
src/trading/ws_client.py

Phase 1 of the trading system: an authenticated Kalshi WebSocket client.

This first cut is deliberately a RAW-FRAME DUMPER. Kalshi's public WS docs
sit behind a redirect wall and can go stale, so rather than code an
order-book parser against documentation we can't verify, we connect for
real and print exactly what the exchange sends. The live frames are ground
truth — the order-book state machine (next file) is built to match them.

It also doubles as the connectivity + credential smoke test: it proves the
WS handshake auth works (same RSA-PSS signing as REST, §7) and whether
there are open markets to subscribe to right now.

Auth: the WS handshake is signed just like a REST GET — sign
`timestamp_ms + "GET" + "/trade-api/ws/v2"` and send the same three
KALSHI-ACCESS-* headers on the upgrade request. Public market-data channels
still require an authenticated handshake (CLAUDE.md §5).

Usage:
    # auto-discover an open market and dump its book/ticker frames
    ./.venv/bin/python -m src.trading.ws_client

    # a specific market, for longer
    ./.venv/bin/python -m src.trading.ws_client --ticker KXMLBGAME-... --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import time

import certifi
import websockets

from src.pipeline.auth import KalshiClient

# Build the TLS context from certifi's CA bundle. Python from python.org on
# macOS doesn't trust the system keychain, so raw asyncio SSL fails cert
# verification (unable to get local issuer) even though `requests` — which
# bundles certifi — works fine. This is the one-line fix.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

WS_PATH = "/trade-api/ws/v2"
WS_HOST = "wss://api.elections.kalshi.com"
REST_HOST = "https://api.elections.kalshi.com"


def _handshake_headers(client: KalshiClient) -> dict[str, str]:
    """The three signed headers for the WS upgrade request. We reuse the
    REST signer, but sign the WS path (not an API-prefixed endpoint)."""
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY": client.key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": client._sign(ts, "GET", WS_PATH),
    }


def _discover_open_market(client: KalshiClient, prefer: str = "KXMLB") -> str | None:
    """Find an ACTIVE open market to subscribe to — highest volume, so it
    actually streams book updates (a random open market can be dead silent).
    Prefer MLB if any MLB market has volume; else the busiest market overall."""
    resp = client.request("GET", "/markets", params={"status": "open", "limit": 1000})
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    if not markets:
        return None
    markets.sort(key=lambda m: m.get("volume", 0) or 0, reverse=True)
    mlb = [m for m in markets if m.get("ticker", "").startswith(prefer)]
    chosen = mlb[0] if (mlb and (mlb[0].get("volume") or 0) > 0) else markets[0]
    print(f"  {len(markets)} open markets; picking busiest "
          f"{prefer if chosen in mlb else ''} market {chosen['ticker']} "
          f"(volume {chosen.get('volume', 0)})")
    return chosen["ticker"]


def _discover_active_markets(client: KalshiClient, n: int = 100) -> list[str]:
    """Return up to n open-market tickers. Volume in the list view is
    unreliable (often 0), so we just take a broad batch and let the WS tell
    us which are actually streaming — subscribing wide is how a real watcher
    catches activity anyway."""
    resp = client.request("GET", "/markets", params={"status": "open", "limit": 1000})
    resp.raise_for_status()
    tickers = [m["ticker"] for m in resp.json().get("markets", [])]
    print(f"  discovered {len(tickers)} open markets; subscribing to {min(n, len(tickers))}")
    return tickers[:n]


def _subscribe_cmd(tickers: list[str], channels) -> dict:
    return {"id": 1, "cmd": "subscribe",
            "params": {"channels": list(channels), "market_tickers": list(tickers)}}


async def stream_session(client: KalshiClient, tickers: list[str],
                         channels=("orderbook_delta", "ticker")):
    """
    One WS session: connect, subscribe to all tickers, and yield parsed
    frames until the connection closes. Deliberately does NOT reconnect —
    the caller owns the reconnect loop, so it can also force a resubscribe on
    a sequence gap (which needs a fresh snapshot, not just a reconnect).
    """
    headers = _handshake_headers(client)
    async with websockets.connect(WS_HOST + WS_PATH, additional_headers=headers,
                                  ssl=_SSL_CTX) as ws:
        # One subscribe PER MARKET. Kalshi's `seq` is per-subscription, so a
        # shared multi-market subscription makes each market see a
        # non-consecutive slice of one global counter — which looks like a
        # constant gap. Separate subscriptions give each market its own sid
        # and its own clean 1,2,3… sequence.
        for i, ticker in enumerate(tickers):
            await ws.send(json.dumps({"id": i + 1, "cmd": "subscribe",
                                      "params": {"channels": list(channels),
                                                 "market_tickers": [ticker]}}))
        async for raw in ws:
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


async def dump(ticker: str, seconds: int) -> None:
    client = KalshiClient.from_env(REST_HOST)
    print(f"connecting to {WS_HOST + WS_PATH} ...")
    count = 0
    start = time.monotonic()
    async for frame in stream_session(client, [ticker]):
        count += 1
        print(f"[{count:>3}] {json.dumps(frame)}")
        if time.monotonic() - start >= seconds:
            break
    print(f"\ndone. received {count} frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", help="Market ticker; auto-discovers if omitted")
    parser.add_argument("--seconds", type=int, default=15, help="How long to listen")
    args = parser.parse_args()

    ticker = args.ticker
    if not ticker:
        client = KalshiClient.from_env(REST_HOST)
        ticker = _discover_open_market(client)
        if not ticker:
            raise SystemExit("No open markets found to subscribe to right now.")

    asyncio.run(dump(ticker, args.seconds))


if __name__ == "__main__":
    main()
