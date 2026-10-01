"""
src/trading/order_book.py

Live L2 order-book state machine for one Kalshi market. Pure logic, no I/O:
it consumes the WS frames (schemas decoded in ws_client.py) and maintains
the current book, so the strategy layer can ask "what's the best bid/ask
and how much size is there right now."

Kalshi's book, and the one non-obvious conversion (verified against the
live ticker in ws_client's dump):

  • The exchange publishes TWO independent ladders — resting YES buy orders
    and resting NO buy orders — each a set of (price, size) levels.
  • A resting YES buy at price q is a YES BID at q (someone will buy YES
    from you at q).
  • A resting NO buy at price p is a YES ASK at (1 − p): buying NO at p is
    economically selling YES at (1 − p), so it's an offer to sell YES to
    you at (1 − p).

  ⇒ best_yes_bid = max(yes ladder prices)
    best_yes_ask = 1 − max(no ladder prices)          # 1 − best NO bid

Prices are the §8 fixed-point DOLLAR strings ("0.4320"); we parse them to
Decimal and never float — decimal fractions can't be represented exactly
in binary, and those errors compound through edge/PnL math.

Sequence integrity: every book frame carries a monotonic `seq`. A snapshot
sets the baseline; each delta must be exactly prev+1. A gap means we missed
a frame and the book is now wrong — apply() returns False so the caller can
resubscribe and get a fresh snapshot.
"""

from __future__ import annotations

from decimal import Decimal

ONE = Decimal("1")


class OrderBook:
    def __init__(self, ticker: str, check_seq: bool = True):
        self.ticker = ticker
        # price(Decimal) -> resting size(Decimal), one dict per ladder.
        self.yes: dict[Decimal, Decimal] = {}
        self.no: dict[Decimal, Decimal] = {}
        self.seq: int | None = None
        self.last_ts_ms: int | None = None
        # Gap detection only makes sense on a SINGLE-market subscription,
        # where `seq` is a clean 1,2,3… sequence. On a multi-market
        # subscription Kalshi's `seq` is a GLOBAL per-connection counter, so
        # each market sees a sparse slice of it and every delta would look
        # like a gap — recorders set check_seq=False and apply unconditionally.
        self.check_seq = check_seq

    # --- frame application ----------------------------------------------------

    def apply(self, frame: dict) -> bool:
        """
        Apply a WS frame. Returns False iff a sequence gap was detected on a
        delta (book is stale → caller must resubscribe). Snapshots always
        succeed and re-baseline the sequence. Non-book frames are ignored.
        """
        typ = frame.get("type")
        msg = frame.get("msg", {})

        if typ == "orderbook_snapshot":
            self._load_snapshot(msg)
            self.seq = frame.get("seq")
            return True

        if typ == "orderbook_delta":
            seq = frame.get("seq")
            if self.check_seq and self.seq is not None and seq != self.seq + 1:
                return False  # gap — book can no longer be trusted
            self._apply_delta(msg)
            self.seq = seq
            self.last_ts_ms = msg.get("ts_ms")
            return True

        return True  # ticker / trade / etc. — not our concern here

    def _load_snapshot(self, msg: dict) -> None:
        self.yes = {Decimal(p): Decimal(s) for p, s in msg.get("yes_dollars_fp", [])}
        self.no = {Decimal(p): Decimal(s) for p, s in msg.get("no_dollars_fp", [])}

    def _apply_delta(self, msg: dict) -> None:
        ladder = self.yes if msg["side"] == "yes" else self.no
        price = Decimal(msg["price_dollars"])
        size = ladder.get(price, Decimal(0)) + Decimal(msg["delta_fp"])
        if size > 0:
            ladder[price] = size
        else:
            ladder.pop(price, None)  # level emptied — drop it

    # --- derived top of book (all in YES terms) -------------------------------

    def best_yes_bid(self) -> Decimal | None:
        return max(self.yes) if self.yes else None

    def best_no_bid(self) -> Decimal | None:
        return max(self.no) if self.no else None

    def best_yes_ask(self) -> Decimal | None:
        nb = self.best_no_bid()
        return (ONE - nb) if nb is not None else None

    def mid(self) -> Decimal | None:
        bid, ask = self.best_yes_bid(), self.best_yes_ask()
        return ((bid + ask) / 2) if (bid is not None and ask is not None) else None

    def spread(self) -> Decimal | None:
        bid, ask = self.best_yes_bid(), self.best_yes_ask()
        return (ask - bid) if (bid is not None and ask is not None) else None

    def yes_bid_size(self) -> Decimal:
        b = self.best_yes_bid()
        return self.yes.get(b, Decimal(0)) if b is not None else Decimal(0)

    def yes_ask_size(self) -> Decimal:
        nb = self.best_no_bid()
        return self.no.get(nb, Decimal(0)) if nb is not None else Decimal(0)

    def top(self) -> dict:
        """Snapshot of the top of book for logging/strategy."""
        return {
            "ticker": self.ticker, "seq": self.seq,
            "yes_bid": self.best_yes_bid(), "yes_ask": self.best_yes_ask(),
            "mid": self.mid(), "spread": self.spread(),
            "bid_size": self.yes_bid_size(), "ask_size": self.yes_ask_size(),
        }


# --- self-test: replay the REAL frames captured live in ws_client -------------
# These are verbatim from the live dump. The assertion proves our derived
# best_yes_ask matches the exchange's own ticker (yes_ask = 1 − best_no_bid),
# so the state machine is correct against ground truth — no live connection
# needed to trust it.

_REPLAY = [
    {"type": "orderbook_snapshot", "seq": 1,
     "msg": {"no_dollars_fp": [["0.4320", "310.00"]]}},
    {"type": "orderbook_delta", "seq": 2,
     "msg": {"price_dollars": "0.4320", "delta_fp": "-310.00", "side": "no", "ts_ms": 1}},
    {"type": "orderbook_delta", "seq": 3,
     "msg": {"price_dollars": "0.4170", "delta_fp": "310.00", "side": "no", "ts_ms": 2}},
    {"type": "orderbook_delta", "seq": 4,
     "msg": {"price_dollars": "0.4170", "delta_fp": "-310.00", "side": "no", "ts_ms": 3}},
    {"type": "orderbook_delta", "seq": 5,
     "msg": {"price_dollars": "0.3950", "delta_fp": "310.00", "side": "no", "ts_ms": 4}},
    # exchange ticker at this point reported yes_ask_dollars = 0.6050
    {"type": "orderbook_delta", "seq": 6,
     "msg": {"price_dollars": "0.3950", "delta_fp": "-310.00", "side": "no", "ts_ms": 5}},
    {"type": "orderbook_delta", "seq": 7,
     "msg": {"price_dollars": "0.4020", "delta_fp": "310.00", "side": "no", "ts_ms": 6}},
    # ticker reported yes_ask_dollars = 0.5980
]


def _selftest() -> None:
    book = OrderBook("REPLAY")
    for frame in _REPLAY[:5]:
        assert book.apply(frame), f"unexpected gap at seq {frame['seq']}"
    assert book.best_yes_ask() == Decimal("0.6050"), book.best_yes_ask()
    print(f"after seq 5: {book.top()}")

    for frame in _REPLAY[5:]:
        assert book.apply(frame)
    assert book.best_yes_ask() == Decimal("0.5980"), book.best_yes_ask()
    print(f"after seq 7: {book.top()}")

    # gap detection: a delta that skips a seq must be rejected.
    assert book.apply({"type": "orderbook_delta", "seq": 99,
                       "msg": {"price_dollars": "0.40", "delta_fp": "10", "side": "no"}}) is False
    print("gap detection: OK")
    print("\nself-test passed — book matches the exchange ticker on live frames.")


if __name__ == "__main__":
    _selftest()
