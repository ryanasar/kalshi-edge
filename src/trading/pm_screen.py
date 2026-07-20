"""
src/trading/pm_screen.py

Rank Polymarket US Liquidity-Incentive markets by how profitable they are for
US to farm — the Polymarket analog of incentive_screen.py (Kalshi). Read-only,
uses the PUBLIC gateway (no auth), so it runs before any API keys exist.

THE ECONOMIC MODEL (why the columns are what they are)
------------------------------------------------------
Polymarket US pays a per-market reward POOL each period for resting limit
orders near the best price. The score for a resting order is (docs, confirmed):

    score = discountFactor ^ (ticks_from_best) × size

and every second a snapshot is taken; each side of the book is independently
normalized to 1.0, so our take on a side is roughly:

    our_share ≈ our_score / (our_score + competitors_score)

Three levers decide whether a market is worth farming — each its own column:

1. POOL RATE ($/day).  `rewardPool` is dollars per PERIOD, but it is a
   PROGRAM-level pool SHARED across every nested market under the same programId
   (e.g. pga_tour_pre_tournament = one $15k pool over 500 golfer markets). So the
   real per-market prize is rewardPool / nested-market-count, then normalized to
   $/day. Treating the pool as per-market overstates the prize by 100-500x.

2. COMPETITION (others' size near best).  The pool splits pro-rata, so our
   share is our_size / (our_size + others_near_best). We measure resting size
   near the touch from the order book; the THIN side is where a quote earns
   outsized share (same edge as Kalshi's lopsided book).

3. ADVERSE SELECTION — and here Polymarket hands us the signal DIRECTLY in the
   discountFactor.  `score = DF^ticks` means DF sets HOW FAR from the touch we
   can rest and still earn:

       safe_ticks(DF) = ln(0.5) / ln(DF)      # ticks back where credit halves

   DF 0.90 → ~6.6 ticks of safe standoff (rest well behind best, barely ever
   filled) → CALM.  DF 0.30 → ~0.6 ticks (must sit AT the touch, picked off) →
   TOXIC. Empirically DF tracks period type: pre_tournament/futures = 0.90
   (calm), live = 0.30 (Polymarket pays the biggest pools here precisely because
   the liquidity gets destroyed). So DF-safety is our ex-ante adverse-selection
   haircut — the §14 lesson, but read off one number instead of guessed.

RANKING objective (risk-adjusted $/day into our account):

    edge = pool_per_day × our_share × safety_mult

HONESTY NOTE.  The exact per-second normalized payout and how unmet Target Size
scales the pool are not fully public, so `our_share`/`edge` are ESTIMATES with a
stated assumption. The RANKING (calm high-DF, thin, decent pool) is robust to
the constant; absolute dollars get calibrated against the first real payout.

Usage:
    ./.venv/bin/python -m src.trading.pm_screen                    # top 25
    ./.venv/bin/python -m src.trading.pm_screen --top 40 --json pm_shortlist.json
    ./.venv/bin/python -m src.trading.pm_screen --capital-per-market 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import httpx

from polymarket_us import PolymarketUS

GATEWAY = "https://gateway.polymarket.us"

# The venue is flooded with microsport churn (Czech table-tennis, LigaMX)
# ordered created-desc, so blindly paginating /v1/incentives never reaches the
# farmable cohort. We instead pull the categories/subcategories that actually
# carry calm or high-DF programs. `category` and `subcategory` filters both work
# server-side (verified); `period`/`status` filters are ignored, so we filter
# those client-side.
CALM_CATEGORIES = ["POL", "MAC", "CUL", "CRY"]           # politics / macro / culture / crypto
SPORT_SUBCATS = ["GOLF", "BASKETBALL", "UFC", "MMA",     # the futures/pre-event cohorts
                 "MOTORSPORT", "TENNIS", "SOCCER", "ESPORTS"]

# Category → drift toxicity. Some underlyings reprice CONTINUOUSLY regardless of
# period type, so a resting quote gets trend-run-over (the Kalshi §5f lesson:
# BTC-vs-gold / crypto drift cost real money). The period model can't see this —
# a crypto "daily" market looks calm but the coin moves 24/7 — so we haircut by
# category. CRY (crypto) is the big one; unknown categories pass at 1.0.
CATEGORY_MULT = {"CRY": 0.15}   # crypto drifts continuously → market-making death

# Period → how much calm runway before the catalyst. DF already prices most of
# the adverse selection; this is a secondary haircut for catalyst proximity, so
# a low-DF "live" market can't sneak through on a big pool alone.
PERIOD_MULT = {
    "futures": 1.00, "pre_tournament": 1.00, "pre_event": 1.00, "preevent": 1.00,
    "tournament_with_cutoff": 0.85,   # spans the event; calm pre, toxic once live
    "daily": 0.80, "early": 0.55, "day_of": 0.45, "live": 0.10,
}

# Rough period length (days) to normalize the pool to $/day.
PERIOD_DAYS = {
    "futures": 30.0, "pre_tournament": 4.0, "tournament_with_cutoff": 4.0,
    "pre_event": 3.0, "preevent": 3.0, "daily": 1.0, "early": 0.5,
    "day_of": 0.5, "live": 0.15,
}


def _get(params: dict, tries: int = 5) -> dict:
    """GET /v1/incentives with retries — the gateway times out intermittently
    from a residential link, and a broad scan must survive one blip."""
    for i in range(tries):
        try:
            return httpx.get(f"{GATEWAY}/v1/incentives", params=params, timeout=12).json()
        except Exception:
            time.sleep(1.0 * (i + 1))
    return {}


def _pull(filt: dict, max_pages: int) -> list[dict]:
    """Paginate one category/subcategory slice (cursor = nextPageToken)."""
    out, tok = [], None
    for _ in range(max_pages):
        params = {**filt, "page_size": 500}
        if tok:
            params["pageToken"] = tok
        r = _get(params)
        out += r.get("programs", [])
        tok = r.get("nextPageToken")
        if not tok:
            break
    return out


def safe_ticks(df: float) -> float:
    """Ticks back from best where reward credit halves = the adverse-selection
    standoff DF buys us. Higher = calmer (rest further from the touch)."""
    if df <= 0 or df >= 1:
        return 0.0
    return math.log(0.5) / math.log(df)


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Candidate:
    market_slug: str
    category: str
    subcategory: str
    program_id: str
    period: str
    program_pool: float        # rewardPool — the WHOLE program's pool
    n_markets: int             # nested markets sharing that pool
    pool: float                # per-market pool = program_pool / n_markets
    pool_per_day: float        # per-market pool normalized to $/day
    discount_factor: float
    safe_ticks: float          # DF-derived safe standoff (adverse-selection quality)
    target_size: float
    price: float               # yes mid (0..1)
    spread_c: float            # ask − bid, cents
    bid_size: float            # resting size at best bid
    ask_size: float            # resting size at best ask
    thin_side: float           # min(bid,ask) — the under-supplied side
    our_size: float            # contracts we'd rest per side
    our_share: float           # estimated share on the thin side
    period_mult: float
    edge_per_day: float        # risk-adjusted $/day — the ranking key


def _best_book(client: PolymarketUS, slug: str) -> tuple[float, float, float, float] | None:
    """(best_bid, best_ask, bid_depth_at_best, ask_depth_at_best) or None if not
    two-sided. Uses the BBO endpoint, NOT book(): the L2 `markets.book()` returns
    None even for markets that are actively trading (verified against
    atc-lmx-pac-que — book()=None but bbo()=0.60/0.61 depth 10/12). bbo() is the
    reliable source for best bid/ask + near-best depth."""
    try:
        d = client.markets.bbo(slug).get("marketData", {})
    except Exception:
        return None
    bid = _num((d.get("bestBid") or {}).get("value"))
    ask = _num((d.get("bestAsk") or {}).get("value"))
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return bid, ask, _num(d.get("bidDepth")), _num(d.get("askDepth"))


def _active_period(prog: dict) -> dict | None:
    act = [t for t in prog.get("timePeriods", []) if t.get("status") == "active"]
    return act[0] if act else None


def screen(*, capital: float, top: int, min_pool_per_day: float,
           pages_per_slice: int, min_price: float, max_price: float,
           max_spread_c: float, enrich_n: int) -> list[Candidate]:
    # 1. gather active programs across the farmable slices
    progs: list[dict] = []
    for cat in CALM_CATEGORIES:
        progs += _pull({"category": cat}, pages_per_slice)
    for sub in SPORT_SUBCATS:
        progs += _pull({"category": "SPR", "subcategory": sub}, pages_per_slice)

    # 1b. rewardPool is PROGRAM-level, shared across every nested market under the
    # same programId (verified: pga_tour_pre_tournament = one $15k pool over 500
    # golfer markets). So the real per-market prize is pool / nested-market-count.
    # Count the nested markets per active programId first.
    n_by_program: dict[str, set] = defaultdict(set)
    for p in progs:
        t = _active_period(p)
        if t:
            n_by_program[t.get("programId", "")].add(p.get("marketSlug", ""))

    # 2. keep active periods, compute the CHEAP per-market economics (no book yet)
    rows: list[dict] = []
    seen: set[str] = set()
    for p in progs:
        t = _active_period(p)
        if not t:
            continue
        slug = p.get("marketSlug", "")
        if slug in seen:
            continue
        seen.add(slug)
        period = t.get("period", "")
        program_pool = _num(t.get("rewardPool"))
        n_markets = max(1, len(n_by_program[t.get("programId", "")]))
        pool = program_pool / n_markets                 # per-market share of the pool
        ppd = pool / PERIOD_DAYS.get(period, 1.0)
        if ppd < min_pool_per_day:
            continue
        rows.append({"p": p, "t": t, "slug": slug, "period": period,
                     "program_pool": program_pool, "n_markets": n_markets,
                     "pool": pool, "ppd": ppd})

    # 3. enrich the top-by-pool-rate with book depth (bounds the slow book calls)
    rows.sort(key=lambda r: r["ppd"], reverse=True)
    client = PolymarketUS()
    cands: list[Candidate] = []
    for r in rows[:enrich_n]:
        bk = _best_book(client, r["slug"])
        if not bk:
            continue
        bid, ask, bsz, asz = bk
        price = (bid + ask) / 2
        spread_c = (ask - bid) * 100
        if not (min_price <= price <= max_price):
            continue
        # A wide book means no real market to rest in (pga_pretournament shows
        # 0.01/0.38 token orders); skip — you can't farm a spread you'd have to
        # cross to trade.
        if spread_c > max_spread_c:
            continue
        df = _num(r["t"].get("discountFactor"))
        target = _num(r["t"].get("targetSize")) or 10000.0
        thin = min(bsz, asz)
        our_size = min(capital, target)          # ~capital contracts, capped at target
        share = our_size / (our_size + thin) if (our_size + thin) > 0 else 0.0
        pmult = PERIOD_MULT.get(r["period"], 0.4)
        # safety scales the edge: a calm high-DF market lets us rest back and hold
        # the share with little fill risk; a DF-0.30 market can't. Normalize
        # safe_ticks(0.9)≈6.6 → ~1.0; clamp so it's a multiplier in (0,1].
        safety = min(1.0, safe_ticks(df) / 6.6)
        cat_mult = CATEGORY_MULT.get(r["p"].get("category", ""), 1.0)
        edge = r["ppd"] * share * pmult * safety * cat_mult
        cands.append(Candidate(
            market_slug=r["slug"], category=r["p"].get("category", ""),
            subcategory=r["p"].get("subcategory", ""),
            program_id=r["t"].get("programId", ""), period=r["period"],
            program_pool=round(r["program_pool"], 2), n_markets=r["n_markets"],
            pool=round(r["pool"], 2), pool_per_day=round(r["ppd"], 2),
            discount_factor=df, safe_ticks=round(safe_ticks(df), 1),
            target_size=target, price=round(price, 3),
            spread_c=round(spread_c, 2),
            bid_size=round(bsz, 0), ask_size=round(asz, 0), thin_side=round(thin, 0),
            our_size=round(our_size, 0), our_share=round(share, 4),
            period_mult=pmult, edge_per_day=round(edge, 3),
        ))
    cands.sort(key=lambda c: c.edge_per_day, reverse=True)
    return cands[:top]


def _print_table(cands: list[Candidate]) -> None:
    hdr = (f"{'#':>2} {'market':<32} {'DF':>4} {'progPool':>8} {'#mkt':>5} "
           f"{'$/mkt/d':>7} {'px':>5} {'sprd':>5} {'thin':>6} {'share':>6} "
           f"{'period':<9} {'edge/d':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(cands, 1):
        print(f"{i:>2} {c.market_slug:<32.32} {c.discount_factor:>4} {c.program_pool:>8.0f} "
              f"{c.n_markets:>5} {c.pool_per_day:>7.2f} {c.price:>5.2f} {c.spread_c:>4.0f}c "
              f"{c.thin_side:>6.0f} {c.our_share:>6.1%} {c.period:<9.9} {c.edge_per_day:>7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital-per-market", type=float, default=500.0,
                    help="$ reserved per market (≈ contracts quoted per side)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-pool-per-day", type=float, default=1.0,
                    help="min PER-MARKET pool $/day (pool is shared across a program)")
    ap.add_argument("--pages-per-slice", type=int, default=2,
                    help="pages (×500) to pull per category/subcategory")
    ap.add_argument("--enrich-n", type=int, default=60,
                    help="how many top-by-pool programs to fetch books for")
    ap.add_argument("--min-price", type=float, default=0.05)
    ap.add_argument("--max-price", type=float, default=0.95)
    ap.add_argument("--max-spread-c", type=float, default=5.0,
                    help="skip markets whose spread exceeds this (no real book to rest in)")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    cands = screen(
        capital=args.capital_per_market, top=args.top,
        min_pool_per_day=args.min_pool_per_day, pages_per_slice=args.pages_per_slice,
        min_price=args.min_price, max_price=args.max_price,
        max_spread_c=args.max_spread_c, enrich_n=args.enrich_n,
    )
    _print_table(cands)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(c) for c in cands], f, indent=1)
        print(f"\nwrote {len(cands)} → {args.json}")


if __name__ == "__main__":
    main()
