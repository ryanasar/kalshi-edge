"""
src/trading/quoter.py

Minimal two-sided quoting loop — the first code that places orders which CAN
fill and lose real money. Purpose: MEASURE the real adverse-selection bleed and
best-price presence of liquidity provision in ONE incentive pool, at tiny size,
before scaling capital or porting to always-on infra.

SAFETY (real money; every knob small, every exit flattens):
  • size/position: quotes `size` contracts/side (default 2), inventory capped at
    ±max_position (default 4). In a $0.10–0.50 market that bounds absolute risk
    to a few dollars even in a total wipeout.
  • bounded runtime: exits after --minutes.
  • max-loss kill: if marked PnL ≤ −max_loss, cancel + flatten + stop.
  • ALWAYS flattens on EVERY exit path (timeout, error, Ctrl-C) via try/finally:
    cancels resting orders AND closes open inventory with a marketable order
    priced through the touch, then verifies flat and SCREAMS if not.
  • only quotes a two-sided book (the subsidy's own requirement anyway).
  • self-tracks position from ITS OWN fills (starts flat, only its orders move
    it), so the cap and flatten never depend on guessing API position fields.

Strategy: join best bid and best ask (be AT best for the 1.0x incentive score),
re-quote when the touch moves, stop quoting a side once it would push inventory
past the cap. Logs each cycle so fills and bleed are visible live.

Usage (REAL MONEY — needs --yes):
    ./.venv/bin/python -m src.trading.quoter --ticker KXWCADS-26JUL19-ABNB \
        --size 2 --max-position 4 --minutes 15 --yes
"""

from __future__ import annotations

import argparse
import time
from decimal import Decimal

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient
from src.trading import orders

PROD_HOST = CANDIDATE_HOSTS[0]
MAKER_RATE = Decimal("0.0175")
MIN_P, MAX_P = Decimal("0.01"), Decimal("0.99")


def maker_fee(p: Decimal, n: int) -> Decimal:
    return MAKER_RATE * p * (1 - p) * n


def _remaining(o: dict) -> Decimal:
    r = o.get("remaining_count") or o.get("remaining_count_fp")
    if r is not None:
        return Decimal(str(r))
    init = Decimal(str(o.get("initial_count_fp", o.get("initial_count", "0"))))
    fill = Decimal(str(o.get("fill_count_fp", o.get("fill_count", "0"))))
    return init - fill


class Quoter:
    def __init__(self, client, ticker, size, max_position, max_loss):
        self.c = client
        self.ticker = ticker
        self.size = size
        self.max_position = max_position
        self.max_loss = Decimal(str(max_loss))
        self.pos = 0                 # net YES contracts (self-tracked from fills)
        self.cash = Decimal(0)       # realized cash from fills
        self.fees = Decimal(0)
        self.fills = 0
        self.bid = None              # {"id","price","rem"} or None
        self.ask = None
        self.cycles = 0
        self.at_best = 0             # cycles resting two-sided at best

    # --- pnl ---------------------------------------------------------------
    def pnl(self, mid: Decimal) -> Decimal:
        return self.cash + Decimal(self.pos) * mid - self.fees

    def _record_fill(self, side: str, price: Decimal, n: Decimal) -> None:
        n_i = int(n)
        if n_i <= 0:
            return
        if side == "bid":                       # bought YES at my resting bid
            self.pos += n_i
            self.cash -= price * n_i
        else:                                   # sold YES at my resting ask
            self.pos -= n_i
            self.cash += price * n_i
        self.fees += maker_fee(price, n_i)
        self.fills += 1
        print(f"    FILL {side} {n_i}@{price}  -> pos={self.pos} cash={self.cash:.4f}")

    # --- fill detection: diff my tracked orders vs the live resting list ----
    def _detect_fills(self) -> None:
        live = {o.get("order_id") or o.get("id"): o for o in orders.resting_orders(self.c)}
        for tag in ("bid", "ask"):
            o = getattr(self, tag)
            if o is None:
                continue
            if o["id"] in live:
                rem = _remaining(live[o["id"]])
                if rem < o["rem"]:
                    self._record_fill(tag, o["price"], o["rem"] - rem)
                    o["rem"] = rem
                if rem <= 0:
                    setattr(self, tag, None)
            else:  # gone and we didn't cancel it -> fully filled (GTC won't expire)
                self._record_fill(tag, o["price"], o["rem"])
                setattr(self, tag, None)

    def _requote_side(self, tag: str, want: bool, price: Decimal | None) -> None:
        o = getattr(self, tag)
        if not want or price is None:
            if o is not None:
                orders.cancel(self.c, o["id"]); setattr(self, tag, None)
            return
        if o is not None and o["price"] == price:
            return  # already at best on this side — leave it (keep queue priority)
        if o is not None:
            orders.cancel(self.c, o["id"]); setattr(self, tag, None)
        try:
            placed = orders.rest_limit(self.c, self.ticker, tag, price,
                                       count=self.size, post_only=True, guard=False)
            oid = placed.get("order_id") or placed.get("id")
            setattr(self, tag, {"id": oid, "price": price, "rem": Decimal(self.size)})
        except orders.OrderError as e:
            print(f"    (skip {tag} @ {price}: {e})")  # post_only rejected a racing quote

    # --- one cycle ---------------------------------------------------------
    def step(self) -> bool:
        """Returns False to stop (kill switch)."""
        self.cycles += 1
        self._detect_fills()
        bid, ask = orders._best_quote(self.c, self.ticker)
        if bid is None or ask is None:
            return True  # one-sided book: don't quote, just wait
        mid = (bid + ask) / 2
        pnl = self.pnl(mid)
        two_sided = self.bid is not None and self.ask is not None \
            and self.bid["price"] == bid and self.ask["price"] == ask
        if two_sided:
            self.at_best += 1
        print(f"  [{self.cycles:>3}] bid {bid} ask {ask} | pos {self.pos} "
              f"pnl {pnl:+.4f} fills {self.fills} atbest {self.at_best}")
        if pnl <= -self.max_loss:
            print(f"  !! KILL: pnl {pnl:.4f} ≤ -{self.max_loss}")
            return False
        self._requote_side("bid", self.pos < self.max_position, bid)
        self._requote_side("ask", self.pos > -self.max_position, ask)
        return True

    # --- flatten everything (called on EVERY exit) -------------------------
    def flatten(self) -> None:
        print("flattening: cancelling all resting orders ...")
        try:
            orders.cancel_all(self.c)
        except Exception as e:  # noqa: BLE001
            print(f"  cancel_all error: {e}")
        self.bid = self.ask = None
        if self.pos != 0:
            bid, ask = orders._best_quote(self.c, self.ticker)
            # price THROUGH the touch to guarantee the close fills (a few ¢ slip
            # on ≤max_position contracts — being flat beats saving pennies).
            if self.pos > 0 and bid is not None:      # long -> sell YES
                px = max(bid - Decimal("0.03"), MIN_P)
                side = "ask"
            elif self.pos < 0 and ask is not None:    # short -> buy YES
                px = min(ask + Decimal("0.03"), MAX_P)
                side = "bid"
            else:
                print(f"  !! cannot price flatten (one-sided book); pos={self.pos} LEFT OPEN")
                return
            print(f"  closing pos={self.pos} via marketable {side} {abs(self.pos)}@{px}")
            try:
                orders.rest_limit(self.c, self.ticker, side, px, count=abs(self.pos),
                                  post_only=False, guard=False)
            except orders.OrderError as e:
                print(f"  !! flatten order FAILED: {e}")
        time.sleep(1.0)
        left = orders.resting_orders(self.c)
        mp = orders.positions(self.c).get("market_positions", [])
        openpos = [p for p in mp if p.get("ticker") == self.ticker]
        if left or openpos:
            print(f"  !! NOT FLAT — resting={len(left)} positions={openpos} — CHECK MANUALLY")
        else:
            print("  verified flat: 0 resting, 0 positions.")

    def summary(self) -> None:
        bid, ask = orders._best_quote(self.c, self.ticker)
        mid = (bid + ask) / 2 if (bid and ask) else Decimal("0.5")
        print(f"\n{'='*56}\nSESSION SUMMARY  {self.ticker}\n{'='*56}")
        print(f"  cycles {self.cycles} | two-sided-at-best {self.at_best} "
              f"({self.at_best/max(self.cycles,1):.0%})")
        print(f"  fills {self.fills} | final pos {self.pos}")
        print(f"  realized cash {self.cash:+.4f} | fees {self.fees:.4f} "
              f"| est PnL {self.pnl(mid):+.4f}")
        print("  (bleed = PnL after flatten; positive best-price% = subsidy-qualifying)\n")


def run(client, ticker, size, max_position, minutes, max_loss, poll) -> None:
    # precondition: start FLAT. Never begin a quoting session with open risk.
    if orders.resting_orders(client):
        raise SystemExit("account has resting orders — cancel-all first.")
    if [p for p in orders.positions(client).get("market_positions", []) if p.get("ticker") == ticker]:
        raise SystemExit(f"already hold a position in {ticker} — flatten first.")

    q = Quoter(client, ticker, size, max_position, max_loss)
    deadline = time.monotonic() + minutes * 60
    print(f"quoting {ticker}  size={size} maxpos=±{max_position} "
          f"maxloss=${max_loss} for {minutes}min\n")
    try:
        while time.monotonic() < deadline:
            if not q.step():
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        q.flatten()      # the load-bearing safety net — runs no matter what
        q.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--size", type=int, default=2, help="contracts per side")
    ap.add_argument("--max-position", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--max-loss", type=float, default=5.0, help="dollars")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between cycles")
    ap.add_argument("--yes", action="store_true", help="required — places REAL orders")
    args = ap.parse_args()
    if not args.yes:
        raise SystemExit("This places REAL, fillable orders. Re-run with --yes.")
    client = KalshiClient.from_env(host=PROD_HOST)
    run(client, args.ticker, args.size, args.max_position,
        args.minutes, args.max_loss, args.poll)


if __name__ == "__main__":
    main()
