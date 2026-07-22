"""
src/trading/orders.py

Order placement + cancellation for Kalshi — the first WRITE capability in the
system (everything before this was read-only: streaming, book state, ticks).
This is the foundation for the Liquidity-Incentive-Program quoting loop, but
it is deliberately just the primitive: create one resting order, cancel one
order, list what's resting, read balance/positions. The loop is built ON this,
later, once this is proven on a single contract.

REAL MONEY. Safety model (every default leans toward "cannot lose money"):
  • post_only=True by DEFAULT — the exchange REJECTS the order if it would
    fill immediately, so a resting order can never accidentally cross and take
    liquidity (paying the taker fee + immediate adverse fill).
  • Prices/counts are FIXED-POINT DOLLAR STRINGS (§8): price "0.1000", count
    "1.00". Never cents, never float. Sending 10¢ as "10" instead of "0.10"
    is exactly the fat-finger that buys at $10 — the format guard makes it
    unrepresentable (price must be in (0,1)).
  • rest_limit refuses a bid at/above the best ask (and an ask at/below the
    best bid) client-side, BEFORE sending — belt-and-suspenders with post_only.
  • Every create returns an order_id; cancel() takes it. We never place an
    order we cannot cancel — cancel is implemented and tested in the same
    breath as create.

Endpoints (V2, on api.elections.kalshi.com per the docs):
  create : POST   /trade-api/v2/portfolio/events/orders
  cancel : DELETE /trade-api/v2/portfolio/orders/{order_id}
  list   : GET    /trade-api/v2/portfolio/orders
  side   : "bid" = buy YES, "ask" = sell YES.

Usage (read-only until funded — a resting bid needs buying power):
    ./.venv/bin/python -m src.trading.orders balance
    ./.venv/bin/python -m src.trading.orders orders
    # the guarded single-contract smoke test (needs a funded account):
    ./.venv/bin/python -m src.trading.orders dry-run --ticker KXMLB... [--yes]
"""

from __future__ import annotations

import argparse
import json
import uuid
from decimal import ROUND_DOWN, Decimal

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient

PROD_HOST = CANDIDATE_HOSTS[0]
CREATE_PATH = "/portfolio/events/orders"          # V2 create
CANCEL_PATH = "/portfolio/events/orders"          # V2 cancel: DELETE {CANCEL_PATH}/{id}
ORDERS_PATH = "/portfolio/orders"                 # list still lives here (V1 list is fine)

ONE = Decimal("1")
MIN_PRICE = Decimal("0.01")   # Kalshi min tick is 1¢
MAX_PRICE = Decimal("0.99")


class OrderError(RuntimeError):
    pass


def _fp_price(price) -> str:
    """Validate a price is a real in-range probability and format it as the
    fixed-point dollar string Kalshi wants ('0.4300'). Guards the cents/dollars
    fat-finger: anything ≥ 1 (i.e. someone passed cents) is rejected."""
    p = Decimal(str(price))
    if not (MIN_PRICE <= p <= MAX_PRICE):
        raise OrderError(f"price {p} out of [0.01, 0.99] — did you pass cents?")
    return f"{p.quantize(Decimal('0.0001'))}"


def _fp_count(count: int) -> str:
    if count < 1:
        raise OrderError(f"count {count} must be ≥ 1")
    return f"{Decimal(count).quantize(Decimal('0.01'))}"


# --- read helpers -----------------------------------------------------------

def balance(client: KalshiClient) -> dict:
    r = client.request("GET", "/portfolio/balance")
    r.raise_for_status()
    return r.json()


def resting_orders(client: KalshiClient) -> list[dict]:
    r = client.request("GET", ORDERS_PATH, params={"status": "resting", "limit": 100})
    r.raise_for_status()
    return r.json().get("orders", [])


def positions(client: KalshiClient) -> dict:
    r = client.request("GET", "/portfolio/positions", params={"limit": 100})
    r.raise_for_status()
    return r.json()


def _best_quote(client: KalshiClient, ticker: str) -> tuple[Decimal | None, Decimal | None]:
    """(best_yes_bid, best_yes_ask) in dollars, for the marketability guard."""
    r = client.request("GET", f"/markets/{ticker}")
    r.raise_for_status()
    m = r.json().get("market", {})
    bid = m.get("yes_bid_dollars")
    ask = m.get("yes_ask_dollars")
    return (Decimal(bid) if bid else None, Decimal(ask) if ask else None)


# --- the write primitives ---------------------------------------------------

def rest_limit(client: KalshiClient, ticker: str, side: str, price, count: int = 1,
               post_only: bool = True, stp: str = "taker_at_cross",
               guard: bool = True) -> dict:
    """Place ONE resting limit order. side='bid' buys YES, 'ask' sells YES.
    Refuses a marketable price client-side (guard) and post_only rejects it
    server-side too. Returns the created order dict (carries order_id)."""
    if side not in ("bid", "ask"):
        raise OrderError(f"side must be 'bid' or 'ask', got {side!r}")
    p = Decimal(_fp_price(price))

    if guard:  # never let the "resting" order actually be marketable
        bid, ask = _best_quote(client, ticker)
        if side == "bid" and ask is not None and p >= ask:
            raise OrderError(f"bid {p} ≥ best ask {ask} — would take, not rest")
        if side == "ask" and bid is not None and p <= bid:
            raise OrderError(f"ask {p} ≤ best bid {bid} — would take, not rest")

    body = {
        "ticker": ticker,
        "side": side,
        "count": _fp_count(count),
        "price": _fp_price(price),
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": stp,
        "post_only": post_only,
        "client_order_id": f"kedge-{uuid.uuid4()}",
    }
    r = client.request("POST", CREATE_PATH, json=body)
    if not r.ok:
        raise OrderError(f"create failed [{r.status_code}]: {r.text}")
    return r.json().get("order", r.json())


def cancel(client: KalshiClient, order_id: str) -> dict:
    r = client.request("DELETE", f"{CANCEL_PATH}/{order_id}")
    if not r.ok:
        raise OrderError(f"cancel failed [{r.status_code}]: {r.text}")
    return r.json()


def cancel_all(client: KalshiClient) -> int:
    """Flatten all resting orders — the manual kill switch."""
    n = 0
    for o in resting_orders(client):
        oid = o.get("order_id") or o.get("id")
        if oid:
            cancel(client, oid)
            n += 1
    return n


# --- the guarded single-contract smoke test ---------------------------------

def dry_run(client: KalshiClient, ticker: str) -> None:
    """Prove create→rest→cancel on ONE contract at a deep non-marketable price.
    Triple safety: post_only, 1 contract, and a price near the 1¢ floor so it
    cannot fill even if post_only were ignored. Needs a funded account."""
    bid, ask = _best_quote(client, ticker)
    print(f"market {ticker}: best_bid={bid} best_ask={ask}")
    test_price = MIN_PRICE  # 1¢ bid — deep, non-marketable, ~1¢ max exposure
    print(f"placing 1-contract post_only BID @ ${test_price} (cannot fill) ...")
    order = rest_limit(client, ticker, "bid", test_price, count=1, post_only=True)
    oid = order.get("order_id") or order.get("id")
    print(f"  created: order_id={oid}\n  {json.dumps(order)[:400]}")

    resting = resting_orders(client)
    here = [o for o in resting if (o.get("order_id") or o.get("id")) == oid]
    print(f"  confirmed resting: {'YES' if here else 'NO'} "
          f"({len(resting)} total resting)")

    if oid:
        print("  cancelling ...")
        cancel(client, oid)
        still = [o for o in resting_orders(client)
                 if (o.get("order_id") or o.get("id")) == oid]
        print(f"  cancelled: {'YES' if not still else 'NO — STILL RESTING, cancel manually'}")
    print("dry-run complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance")
    sub.add_parser("orders")
    sub.add_parser("positions")
    sub.add_parser("cancel-all")
    dr = sub.add_parser("dry-run")
    dr.add_argument("--ticker", required=True)
    dr.add_argument("--yes", action="store_true", help="required to actually place")
    args = ap.parse_args()

    client = KalshiClient.from_env(host=PROD_HOST)
    if args.cmd == "balance":
        print(json.dumps(balance(client), indent=1))
    elif args.cmd == "orders":
        os_ = resting_orders(client)
        print(f"{len(os_)} resting orders")
        for o in os_:
            print(f"  {o.get('order_id') or o.get('id')}  {o.get('ticker')}  "
                  f"{o.get('side') or o.get('book_side')}  "
                  f"{o.get('price_dollars', o.get('yes_price'))}  x{o.get('initial_count_fp')}")
    elif args.cmd == "positions":
        print(json.dumps(positions(client), indent=1))
    elif args.cmd == "cancel-all":
        print(f"cancelled {cancel_all(client)} resting orders")
    elif args.cmd == "dry-run":
        if not args.yes:
            raise SystemExit("dry-run places a REAL 1-contract order. Re-run with --yes.")
        dry_run(client, args.ticker)


if __name__ == "__main__":
    main()
