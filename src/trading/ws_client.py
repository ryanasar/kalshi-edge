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
    """Find a currently-open market to subscribe to. Prefer MLB; fall back
    to any open market so we can still exercise the WS plumbing off-season."""
    resp = client.request("GET", "/markets", params={"status": "open", "limit": 200})
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    if not markets:
        return None
    mlb = [m for m in markets if m.get("ticker", "").startswith(prefer)]
    chosen = (mlb or markets)[0]
    print(f"  discovered {len(markets)} open markets "
          f"({len(mlb)} {prefer}*); using {chosen['ticker']}")
    return chosen["ticker"]


async def dump(ticker: str, seconds: int) -> None:
    client = KalshiClient.from_env(REST_HOST)
    headers = _handshake_headers(client)
    uri = WS_HOST + WS_PATH

    print(f"connecting to {uri} ...")
    async with websockets.connect(uri, additional_headers=headers,
                                  ssl=_SSL_CTX) as ws:
        print("  connected. subscribing to orderbook_delta + ticker ...")
        sub = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "ticker"],
                "market_tickers": [ticker],
            },
        }
        await ws.send(json.dumps(sub))

        deadline = asyncio.get_event_loop().time() + seconds
        count = 0
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - asyncio.get_event_loop().time())
            except asyncio.TimeoutError:
                break
            count += 1
            # Pretty-print so we can read the schema off the wire.
            try:
                print(f"[{count:>3}] {json.dumps(json.loads(raw))}")
            except (json.JSONDecodeError, TypeError):
                print(f"[{count:>3}] {raw!r}")
        print(f"\ndone. received {count} frames in {seconds}s.")


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
