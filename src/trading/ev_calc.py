"""
src/trading/ev_calc.py

Expected-value of subsidy farming as a function of how much we SPEND, for a
chosen set of markets. Answers: "if I put $X in, what do I get back?" — and,
because the answer is nonlinear, where the money stops helping.

THE MODEL
---------
For one market, per day:

    subsidy/day ≈ pool/day × discount × share(S)

  • pool/day   — the reward pool, period-normalized to a daily rate
                 (period_reward/1000 ÷ period_hours × 24).
  • discount   — Kalshi's discount_factor_bps (≈0.50), applied as a haircut.
  • share(S)   — our cut of the pool given we rest S contracts at best. Share is
                 S / (S + competition), so it SATURATES: the first dollars buy a
                 lot of share, later dollars buy almost none once S ≫ competition.
                 You can never earn more than `pool/day × discount` in one market,
                 no matter how much you spend — the ceiling every curve bends toward.

Capital → size: a two-sided quote of S contracts/side reserves S×price on the
bid and S×(1−price) on the ask = **S dollars total**, so $X of capital ≈ S = X
contracts. That capital is also the worst-case at-risk amount if one side fully
fills (why resting S requires maxPosition ≥ S).

THE REWARD-MODEL BRACKET (the load-bearing unknown)
---------------------------------------------------
The exact LIP formula isn't public — specifically, how a two-sided quote's two
sides combine. We don't guess; we BRACKET it, the same way the backtest brackets
fills (CLAUDE.md §6):

  • OPT  (per-side pools): each side is rewarded from its own pool, so being big
         on the THIN side captures that side's pool; the fat side adds little.
         share = S / (S + competition_thin_side).   ← thin side is the prize
  • CONS (one pooled reward over summed size): our contribution competes against
         ALL resting size on BOTH sides.
         share = 2S / (2S + competition_bid + competition_ask).  ← fat side drowns us

TOM being green on a thin ask side is evidence reality is closer to OPT than
CONS, but we report both. Trading P&L (spread capture − maker fees − adverse
selection) is treated as ~0 for STABLE markets we pull before their catalyst
(TOM currently runs slightly positive on it) — so EV ≈ subsidy. This is a
subsidy model, not a promise; absolute dollars calibrate at the first payout.

RUNWAY: a market earns until it resolves OR the program ends (Sept 1 2026),
whichever comes first. Life-EV = subsidy/day × farmable_days.

Usage:
    ./.venv/bin/python -m src.trading.ev_calc               # the default picks
    ./.venv/bin/python -m src.trading.ev_calc KXUSPPIYOY-26AUG13-T5.6 KXUSEDCARCPI-26AUG12-T179.25
    ./.venv/bin/python -m src.trading.ev_calc --spends 200,500,1000,2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient

PROGRAM_END = datetime(2026, 9, 1, tzinfo=timezone.utc)  # LIP ends — hard runway cap
BAND_CENTS = 3
CHUNK = 5.0   # $ granularity of the greedy allocator

DEFAULT_TICKERS = [
    "KXUSPPIYOY-26AUG13-T5.6",
    "KXUSEDCARCPI-26AUG12-T179.25",
    "KXUSRETAIL-26AUG14-T1.0",
    "KXBUILDPERMS-26AUG18-T1.500",
    "KXCPINDEX-26AUG12-T333.6",
]


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Market:
    ticker: str
    pool: float         # $ for THIS reward period (period_reward/1000) — fixed, not renewing
    period_h: float     # length of this reward period
    hours_now: float    # hours left in THIS period = how long we can still farm it
    renewals: float     # ≈ how many more same-length windows fit before resolution/Sept-1
    discount: float
    comp_thin: float    # competing size near best on the thinner side
    comp_bid: float     # near-best size on the yes-bid side
    comp_ask: float     # near-best size on the yes-ask side
    target: float
    price: float

    def _share(self, S: float, model: str) -> float:
        S = min(S, self.target)                            # qualifying size caps at target
        if model == "opt":
            return S / (S + self.comp_thin) if (S + self.comp_thin) > 0 else 0.0
        denom = 2 * S + self.comp_bid + self.comp_ask      # cons: fat side drowns us
        return (2 * S) / denom if denom > 0 else 0.0

    # Subsidy we can BANK in the current window: reward accrues at pool/period_h
    # per hour, we capture our share for the hours_now we're present.
    def ev_window(self, S: float, model: str) -> float:
        pool_per_hr = self.pool / self.period_h
        return pool_per_hr * self.discount * self._share(S, model) * self.hours_now

    # Upside IF the incentive renews at the same size/competition each window
    # until the market resolves (or Sept 1). An ASSUMPTION, reported separately.
    def ev_if_renews(self, S: float, model: str) -> float:
        return self.ev_window(S, model) * max(1.0, self.renewals)


def _band(levels) -> float:
    if not levels:
        return 0.0
    best = _num(levels[-1][0])  # Kalshi lists levels ascending — best bid is LAST
    return sum(_num(sz) for px, sz in levels if abs(_num(px) - best) * 100 <= BAND_CENTS)


def load_programs(client: KalshiClient) -> dict[str, dict]:
    """market_ticker → its active liquidity program (start ≤ now < end)."""
    now, out, cursor, pages = datetime.now(timezone.utc), {}, None, 0
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = client.request("GET", "/incentive_programs", params=params)
        r.raise_for_status()
        d = r.json()
        for p in d.get("incentive_programs", []):
            if p.get("incentive_type") != "liquidity" or p.get("paid_out"):
                continue
            s, e = _iso(p.get("start_date")), _iso(p.get("end_date"))
            if s and e and s <= now < e:
                out[p["market_ticker"]] = p
        cursor = d.get("next_cursor")
        pages += 1
        if not cursor or pages > 60:
            break
    return out


def build_market(client: KalshiClient, ticker: str, prog: dict) -> Market | None:
    m = client.request("GET", f"/markets/{ticker}").json().get("market", {})
    bid, ask = _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
    if bid <= 0 or ask <= 0:
        return None
    ob = client.request("GET", f"/markets/{ticker}/orderbook", params={"depth": 40}) \
        .json().get("orderbook_fp", {})
    comp_bid = _band(ob.get("yes_dollars"))          # yes-bid side
    comp_ask = _band(ob.get("no_dollars"))           # yes-ask side (= NO bids)

    s, e = _iso(prog["start_date"]), _iso(prog["end_date"])
    now = datetime.now(timezone.utc)
    period_h = max((e - s).total_seconds() / 3600, 1e-6)
    hours_now = max((e - now).total_seconds() / 3600, 0.0)   # THIS window's remaining hours
    discount = (_num(prog.get("discount_factor_bps")) / 10000) or 0.5

    # farmable horizon for renewals = to resolution, capped at program end (Sept 1)
    close = _iso(m.get("close_time")) or e
    horizon_h = max((min(close, PROGRAM_END) - now).total_seconds() / 3600, 0.0)
    renewals = horizon_h / period_h if period_h > 0 else 1.0

    return Market(
        ticker=ticker, pool=_num(prog["period_reward"]) / 1000, period_h=period_h,
        hours_now=hours_now, renewals=renewals, discount=discount,
        comp_thin=min(comp_bid, comp_ask), comp_bid=comp_bid, comp_ask=comp_ask,
        target=_num(prog.get("target_size_fp")) or 1000.0, price=(bid + ask) / 2,
    )


def allocate(markets: list[Market], budget: float, model: str) -> dict[str, float]:
    """Greedy water-filling: hand out the budget in $CHUNK increments, each to
    whichever market gains the most banked-window EV from the next chunk. This is
    the EV-maximizing split and naturally favors thin + rich markets, and stops
    feeding a market once its share saturates."""
    alloc = {m.ticker: 0.0 for m in markets}
    spent = 0.0
    while spent + CHUNK <= budget + 1e-9:
        best_m, best_gain = None, -1.0
        for m in markets:
            gain = m.ev_window(alloc[m.ticker] + CHUNK, model) - m.ev_window(alloc[m.ticker], model)
            if gain > best_gain:
                best_gain, best_m = gain, m
        if best_m is None or best_gain <= 0:
            break   # every market saturated — more money does nothing
        alloc[best_m.ticker] += CHUNK
        spent += CHUNK
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--spends", default="150,300,600,1200,2400",
                    help="comma-separated total-spend levels to evaluate")
    args = ap.parse_args()
    tickers = args.tickers or DEFAULT_TICKERS
    spends = [float(x) for x in args.spends.split(",")]

    client = KalshiClient.from_env(host=CANDIDATE_HOSTS[0])
    progs = load_programs(client)

    markets: list[Market] = []
    for tk in tickers:
        if tk not in progs:
            print(f"  ! {tk}: no active liquidity program — skipping")
            continue
        m = build_market(client, tk, progs[tk])
        if m:
            markets.append(m)
    if not markets:
        raise SystemExit("no farmable markets")

    # --- per-market fundamentals ---
    print("\nPER-MARKET (competition = size within 3¢ of best, per side):")
    print(f"{'ticker':<32}{'pool':>6}{'periodH':>8}{'hrsLeft':>8}{'renew':>6}"
          f"{'bidComp':>8}{'askComp':>8}{'thin':>6}{'ceiling':>8}")
    for m in markets:
        print(f"{m.ticker:<32}{m.pool:>6.0f}{m.period_h:>8.0f}{m.hours_now:>8.1f}"
              f"{m.renewals:>6.1f}{m.comp_bid:>8.0f}{m.comp_ask:>8.0f}{m.comp_thin:>6.0f}"
              f"{m.pool*m.discount*m.hours_now/m.period_h:>8.2f}")
    print("  ceiling = most we could earn THIS window (100% share × discount).")

    # --- EV vs spend, bracketed OPT / CONS, EV-optimal allocation ---
    for model, label in (("opt", "OPT  (per-side pools — thin side wins)"),
                          ("cons", "CONS (one pool, summed both-side size)")):
        print(f"\n=== {label} ===")
        print(f"{'spend':>7}{'EV-window':>10}{'ROI%':>7}{'EV-if-renews':>13}   allocation")
        for B in spends:
            alloc = allocate(markets, B, model)
            evw = sum(m.ev_window(alloc[m.ticker], model) for m in markets)
            evr = sum(m.ev_if_renews(alloc[m.ticker], model) for m in markets)
            used = sum(alloc.values())
            roi = (evw / used * 100) if used else 0.0
            top = ", ".join(f"{tk.split('-')[0][2:]}:${a:.0f}"
                            for tk, a in sorted(alloc.items(), key=lambda x: -x[1]) if a > 0)
            print(f"{B:>7.0f}{evw:>10.2f}{roi:>6.0f}%{evr:>13.0f}   {top}")
    print("\nEV-window   = subsidy bankable in the CURRENT reward period (ends ~Jul 19).")
    print("EV-if-renews = IF the incentive re-posts at the same size each period to")
    print("              resolution (an assumption — weekly re-competed, not guaranteed).")
    print("ROI% is EV-window ÷ capital for THIS window (a few days), not annualized.")
    print("Trading P&L assumed ~0 (stable, pulled before release). Absolute $ calibrate")
    print("at the first TOM payout (~Jul 20); OPT vs CONS brackets the unknown formula.")


if __name__ == "__main__":
    main()
