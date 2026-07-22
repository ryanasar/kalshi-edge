"""
src/trading/incentive_screen.py

Rank Kalshi's Liquidity-Incentive-Program (LIP) markets by how profitable it
is for US to farm the subsidy — the input to running the quoter on MORE than
one market at once, but only the ones worth the capital.

THE ECONOMIC MODEL (why the columns are what they are)
------------------------------------------------------
Kalshi pays a per-market reward POOL each PERIOD for resting two-sided
liquidity near the best bid/ask. Our take from a pool is, roughly:

    our_subsidy ≈ pool × discount × (our_qualifying_size / total_qualifying_size)

so three independent levers decide whether a market is worth farming, and the
screen surfaces each one as its own column instead of hiding them in one score:

1. POOL RATE ($/hr).  `period_reward` is in tenths-of-a-cent (÷1000 = dollars),
   confirmed against the live Rewards page ($200 Austin-temp pool ↔ 200000).
   Periods are NOT uniform: many are 1 hour, others 31h / 80h / 127h / 295h.
   A $200/1h pool pays 6× the RATE of a $1000/31h pool, so we normalize every
   pool to $/hour. This is the size of the prize.

2. COMPETITION (others' resting size).  The pool splits pro-rata across every
   maker near the touch, so our share is `our_size / (our_size + others_size)`.
   `others_size` = size already resting at the best bid+ask (from the market's
   top-of-book `*_size_fp` fields; `--deep` sums a wider band from the book).
   A fat pool with a crowded book pays us almost nothing; a thin book is the
   whole opportunity (the live TOM run got ~26% share exactly because its ask
   side was under-supplied). LOW competition is what we hunt.

3. ADVERSE SELECTION (toxicity).  A resting quote fills preferentially when the
   market moves against us (CLAUDE.md §14 — the finding that killed directional
   market-making). Subsidy is only net-positive if the market DOESN'T reprice
   violently while we rest. So we classify each market by how shock-prone its
   underlying is — a scheduled CPI print that sits flat until release (STABLE)
   is farmable; "what the announcers say" DURING the World Cup final (LIVE) is
   a trap. We pull before the catalyst; toxicity says how sharp that catalyst
   is and how little room for error we have.

EXPECTED SUBSIDY (the ranking objective) combines 1–3:

    sub_per_hr = pool_per_hr × discount × our_share × toxicity_multiplier
    roi_per_hr = sub_per_hr / capital_reserved     (capital ≈ our_size dollars,
                 because a two-sided quote reserves size×p + size×(1−p) = size)

`sub_per_hr` is expected dollars/hour INTO our account, risk-adjusted — the
right thing to maximize when allocating a fixed $147 across markets. `roi_per_hr`
is the capital-efficiency tiebreaker.

HONESTY NOTE.  The exact LIP payout formula (how sub-target participation and
the 50% `discount_factor_bps` interact) is not public — it is the open
experiment resolving at the first pool payout (~Jul 19-20, see memory
kalshi-quoter-live). So `our_share`/`sub_per_hr` are ESTIMATES with a stated
assumption, not promises. The RANKING (big thin non-toxic pools first) is
robust to the exact constant; the absolute dollars are calibrated once the
first payout lands.

Usage:
    ./.venv/bin/python -m src.trading.incentive_screen                 # top 25
    ./.venv/bin/python -m src.trading.incentive_screen --top 40 --deep
    ./.venv/bin/python -m src.trading.incentive_screen --capital-per-market 40 \
        --include-toxic --min-hours-left 6 --json shortlist.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal

from src.pipeline.auth import CANDIDATE_HOSTS, KalshiClient

# --- reward-unit + share constants ------------------------------------------
REWARD_PER_DOLLAR = 1000          # period_reward is tenths-of-a-cent → ÷1000 = $
BAND_CENTS = 3                    # "near best" reward band we count competition in (--deep)
DEFAULT_CAPITAL_PER_MARKET = 40.0  # $ reserved per market (≈ contracts, two-sided)


# --- adverse-selection / toxicity classification ----------------------------
# Maps a market to how sharply its fair value can jump WHILE we're resting.
# We can only see the ticker's series prefix and the market title, so the
# classifier is keyword-based and deliberately conservative (unknown → MILD,
# never STABLE). Each tier carries a multiplier that HAIRCUTS expected subsidy
# for the risk that adverse fills eat it — the §14 lesson, applied ex-ante.
TOXICITY = {
    "STABLE": 1.00,   # sits flat until a scheduled, known catalyst we pull before
    "MILD":   0.65,   # drifts but no single violent repricing in a short window
    "EVENT":  0.30,   # jumps discretely when a result is announced/revealed
    "LIVE":   0.10,   # reprices continuously during a live event — the trap
}

# Substring → tier. Checked against UPPERCASED "ticker title". First hit wins,
# so order matters: most-toxic patterns first. Patterns are matched against BOTH
# the ticker (series prefix, e.g. KXAQICITY) and the title (natural language,
# e.g. "the AQI") because Kalshi splits the same concept across the two — the v1
# bug that let AQI/starters through as MILD.
_TOX_RULES: list[tuple[str, str]] = [
    # live in-event repricing (reprices continuously while an event is happening)
    ("WCMENTION", "LIVE"), ("MENTION", "LIVE"), ("HALFTIME", "LIVE"),
    ("KXWCSTART", "LIVE"), ("STARTERS", "LIVE"), ("START FOR", "LIVE"),
    ("WEATHER DELAY", "LIVE"), ("FIRST SONG", "LIVE"), ("CONCEDE", "LIVE"),
    ("GOAL", "LIVE"),
    # same-day weather / air-quality: resolves against a live sensor reading
    ("KXTEMP", "LIVE"), ("KXHIGH", "LIVE"), ("TEMPERATURE IN", "LIVE"),
    ("KXAQICITY", "LIVE"), ("AQI", "LIVE"), ("AIR QUALITY", "LIVE"),
    # discrete reveal / result-announcement jumps (one big step when it lands)
    ("KXWCATTEND", "EVENT"), ("ATTEND", "EVENT"), ("KXWCADS", "EVENT"),
    ("ADVERTISE", "EVENT"), ("HURCAT", "EVENT"), ("CATEGORY", "EVENT"),
    ("CAST", "EVENT"), ("SONGS", "EVENT"), ("WINNER", "EVENT"),
    ("NOMINEE", "EVENT"), ("PRIMARY", "EVENT"), ("ELIMINATION", "EVENT"),
    ("DEBUT DATE", "EVENT"), ("METACRITIC", "EVENT"), ("ROTTEN TOMATOES", "EVENT"),
    ("FEATURED ON", "EVENT"), ("ELECTION", "EVENT"), ("RATE DECISION", "EVENT"),
    # news/entertainment/political jumps that leaked as MILD in the first passes
    ("KXEOWEEK", "EVENT"), ("EXECUTIVE ORDER", "EVENT"), ("KXLOVEISLAND", "EVENT"),
    ("LOVE ISLAND", "EVENT"), ("KXDWTS", "EVENT"), ("KXNBATEAMANNOUNCE", "EVENT"),
    ("KXNBANEXTTEAMCONF", "EVENT"), ("JOINING", "EVENT"), ("KXAPRPOTUS", "EVENT"),
    ("APPROVAL", "EVENT"), ("KXGENERICBALLOT", "EVENT"), ("GENERIC BALLOT", "EVENT"),
    ("VOTEHUB", "EVENT"), ("KXROLEINPRODUCTION", "EVENT"), ("PERFORM AS", "EVENT"),
    ("ROLE IN", "EVENT"), ("NEXT CONFERENCE", "EVENT"),
    # ACCUMULATING-COUNT markets (§5f DRIFT): a running count of discrete events
    # over an OPEN window trends up continuously → market-making death. Match on
    # count NOUNS/tickers, NOT bare "at least"/"between" (those also match
    # scheduled single-VALUE data ranges like "unemployment at least 4%"). The
    # discriminator is "counting events" vs "one measured value".
    ("HOW MANY", "EVENT"), ("ENDORSE", "EVENT"), ("TRUTH SOCIAL", "EVENT"),
    ("TRUTH POSTS", "EVENT"), ("POSTS THIS", "EVENT"), ("TWEET", "EVENT"),
    ("PHOTOGRAPHED", "EVENT"), ("PRESIDENTIAL ACTION", "EVENT"),
    ("KXTRUMPACT", "EVENT"), ("KXTRUMPENDORSE", "EVENT"), ("KXTRUTHSOCIAL", "EVENT"),
    ("KXTRUMPPHOTO", "EVENT"), ("KXTRUMPMENTION", "EVENT"), ("KXTRUMPTRUTH", "EVENT"),
    # scheduled macro/data prints — flat until a known release time (farmable;
    # the resolution-proximity multiplier separately haircuts the endgame).
    ("KXCPI", "STABLE"), ("KXPPI", "STABLE"), ("KXCPICORE", "STABLE"),
    ("CPI", "STABLE"), ("PPI", "STABLE"), ("GAS PRICE", "STABLE"),
    ("KXAAAGASD", "STABLE"), ("HOME SALES", "STABLE"), ("HOUSING START", "STABLE"),
    ("BUILDING PERMIT", "STABLE"), ("TREASURY YIELD", "STABLE"),
    ("RETAIL SALES", "STABLE"), ("JOBLESS", "STABLE"), ("UNEMPLOYMENT", "STABLE"),
    ("PRICE IN JULY", "STABLE"), ("HOURLY PRICE", "STABLE"), ("NET WORTH", "STABLE"),
    ("FEAR & GREED", "STABLE"),
    # DRIFTING markets (§5f) — continuously reprice with live/accumulating data,
    # so a resting quote gets trend-run-over (market-making death). Treat as toxic.
    ("KXUSFLYCAN", "EVENT"), ("CANCELLATIONS", "EVENT"), ("KXBTCVSGOLD", "EVENT"),
    ("OUTPERFORM", "EVENT"), ("VS. GOLD", "EVENT"),
    # crypto/martingale — drifts continuously; MM loses to the walk
    ("BITCOIN", "EVENT"), ("BTC", "EVENT"), ("DXY", "MILD"),
]


def classify_toxicity(ticker: str, title: str) -> str:
    hay = f"{ticker} {title}".upper()
    for needle, tier in _TOX_RULES:
        if needle in hay:
            return tier
    return "MILD"  # unknown underlying → assume it can move, but not violently


def proximity_mult(hours_to_close: float) -> float:
    """Haircut for how close the RESOLUTION catalyst is. Independent of the
    toxicity TYPE: even a STABLE gas-price market is dangerous to rest in during
    its final hours, because as the print approaches the price drifts to 0/1 and
    informed flow picks us off. We farm the calm middle and pull before the end;
    this multiplier prices "how much calm runway is left."
        ≥48h → 1.00 (days of stable farming ahead)
        12-48 → 0.80
        4-12  → 0.50 (catalyst getting close — short farm, quicker exit)
        <4    → 0.20 (endgame; avoid unless the pool is enormous)
    """
    if hours_to_close >= 48:
        return 1.00
    if hours_to_close >= 12:
        return 0.80
    if hours_to_close >= 4:
        return 0.50
    return 0.20


# --- helpers ----------------------------------------------------------------
def _get(client: KalshiClient, endpoint: str, params: dict | None = None, tries: int = 3):
    """GET with retries — the local link to Kalshi times out intermittently, and
    a broad scan shouldn't die on one transient blip. Returns the Response or
    None if every attempt failed."""
    import time as _t
    for i in range(tries):
        try:
            return client.request("GET", endpoint, params=params or {})
        except Exception:
            if i == tries - 1:
                return None
            _t.sleep(1.0 * (i + 1))
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Candidate:
    ticker: str
    event: str            # event_ticker — correlated strikes share one
    title: str
    pool: float           # $ per period
    pool_per_hr: float    # $ per hour (period-normalized)
    period_h: float
    hours_left: float     # until the reward period ends
    hours_to_close: float # until the MARKET resolves (catalyst)
    target: float
    discount: float       # discount_factor as a fraction (0.50)
    price: float          # yes mid, dollars
    spread: float         # yes ask − yes bid, cents
    bid_side: float       # resting size near best on the YES-bid (buy) side
    ask_side: float       # resting size near best on the YES-ask (sell) side
    thin_side: float      # min(bid_side, ask_side) — the under-supplied side we farm
    lopsided: float       # max/min side ratio — how one-sided the book is
    our_size: float       # contracts we'd rest per side
    our_share: float      # estimated fraction we'd earn ON THE THIN SIDE
    toxicity: str
    tox_mult: float
    prox_mult: float      # resolution-proximity haircut
    sub_per_hr: float     # expected $ / hour to us, risk-adjusted
    sub_per_day: float    # capped by hours_left
    roi_per_hr: float     # sub_per_hr / capital, capital-efficiency tiebreak
    score: float          # ranking key = sub_per_hr (already risk-adjusted)
    open_interest: float  # OI — liquidity to unwind against (concentration input)


# --- program ingestion ------------------------------------------------------
def fetch_active_programs(client: KalshiClient) -> list[dict]:
    """All liquidity programs whose reward period is CURRENTLY live (start ≤ now
    < end) and not yet paid out — the only ones we can still earn from."""
    now, out, cursor, pages = _now(), [], None, 0
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = _get(client, "/incentive_programs", params)
        if r is None or not r.ok:
            break
        d = r.json()
        for p in d.get("incentive_programs", []):
            if p.get("incentive_type") != "liquidity" or p.get("paid_out"):
                continue
            s, e = _iso(p.get("start_date")), _iso(p.get("end_date"))
            if not s or not e or not (s <= now < e):
                continue
            out.append(p)
        cursor = d.get("next_cursor")
        pages += 1
        if not cursor or pages > 60:
            break
    return out


def _band_sides(client: KalshiClient, ticker: str) -> tuple[float, float]:
    """Deeper competition proxy: resting size within BAND_CENTS of best, returned
    PER SIDE — (yes-bid side, yes-ask side). Per-side is the point: a market can
    be crowded overall yet have one thin side we can dominate (the TOM book was
    180k on the bid vs 1.1k on the ask). Only used with --deep (extra call)."""
    r = _get(client, f"/markets/{ticker}/orderbook", {"depth": 30})
    if r is None or not r.ok:
        return 0.0, 0.0

    def band(levels) -> float:
        if not levels:
            return 0.0
        # Kalshi returns levels in ASCENDING price order, so the BEST bid is the
        # LAST element, not the first. (Reading levels[0] was the bug that made
        # every market look crowded — it measured depth around the WORST price.)
        best = _num(levels[-1][0])
        return sum(_num(sz) for px, sz in levels if abs(_num(px) - best) * 100 <= BAND_CENTS)

    ob = r.json().get("orderbook_fp", {})
    # yes_dollars = resting YES bids (the buy side); no_dollars = resting NO bids,
    # which is the YES-ask (sell) side. That mapping is the book convention
    # `yes_ask = 1 − best_no_bid` from order_book.py.
    return band(ob.get("yes_dollars")), band(ob.get("no_dollars"))


def enrich(client: KalshiClient, prog: dict, capital: float, deep: bool,
           pause: float) -> Candidate | None:
    """Join one program to its live market and compute the economics. Returns
    None if the market is untradeable right now (no two-sided quote / closed)."""
    ticker = prog["market_ticker"]
    r = _get(client, f"/markets/{ticker}")
    if r is None or not r.ok:
        return None
    m = r.json().get("market", {})
    if m.get("status") != "active":
        return None
    bid = _num(m.get("yes_bid_dollars"))
    ask = _num(m.get("yes_ask_dollars"))
    if bid <= 0 or ask <= 0 or ask <= bid:   # need a real two-sided book to join
        return None

    pool = _num(prog.get("period_reward")) / REWARD_PER_DOLLAR
    s, e = _iso(prog["start_date"]), _iso(prog["end_date"])
    period_h = max((e - s).total_seconds() / 3600, 1e-6)
    hours_left = max((e - _now()).total_seconds() / 3600, 0.0)
    pool_per_hr = pool / period_h
    discount = _num(prog.get("discount_factor_bps")) / 10000 or 0.5
    target = _num(prog.get("target_size_fp")) or 1000.0

    close = _iso(m.get("close_time"))
    hours_to_close = max((close - _now()).total_seconds() / 3600, 0.0) if close else hours_left
    prox = proximity_mult(hours_to_close)

    price = (bid + ask) / 2
    spread_c = round((ask - bid) * 100, 1)

    # competition, PER SIDE: touch size (cheap) or a wider band (--deep). The
    # under-supplied side is where a two-sided quote earns outsized share, so we
    # measure both and key off the thinner one — the TOM edge, made a metric.
    if deep:
        bid_side, ask_side = _band_sides(client, ticker)
        if pause:
            time.sleep(pause)
    else:
        bid_side = _num(m.get("yes_bid_size_fp"))
        ask_side = _num(m.get("yes_ask_size_fp"))
    thin_side = min(bid_side, ask_side)
    lopsided = (max(bid_side, ask_side) / thin_side) if thin_side > 0 else 999.0

    # our participation: `capital` dollars two-sided ≈ `capital` contracts/side
    # (bid reserves size×p, ask reserves size×(1−p); sum = size). Capped at target.
    # Share is estimated on the THIN side — that's the side our size actually
    # moves, and (per the LIP) the binding constraint on a two-sided quote.
    our_size = min(capital, target)
    our_share = our_size / (our_size + thin_side) if (our_size + thin_side) > 0 else 0.0

    tox = classify_toxicity(ticker, m.get("title", ""))
    tox_mult = TOXICITY[tox]

    sub_per_hr = pool_per_hr * discount * our_share * tox_mult * prox
    sub_per_day = sub_per_hr * min(24.0, hours_left)
    roi_per_hr = sub_per_hr / capital if capital > 0 else 0.0

    return Candidate(
        ticker=ticker, event=m.get("event_ticker", ""),
        title=m.get("title", "")[:60], pool=round(pool, 2),
        pool_per_hr=round(pool_per_hr, 3), period_h=round(period_h, 1),
        hours_left=round(hours_left, 1), hours_to_close=round(hours_to_close, 1),
        target=target, discount=discount, price=round(price, 3), spread=spread_c,
        bid_side=round(bid_side, 0), ask_side=round(ask_side, 0),
        thin_side=round(thin_side, 0), lopsided=round(lopsided, 1),
        open_interest=_num(m.get("open_interest_fp")),
        our_size=round(our_size, 1),
        our_share=round(our_share, 4), toxicity=tox, tox_mult=tox_mult,
        prox_mult=prox, sub_per_hr=round(sub_per_hr, 4),
        sub_per_day=round(sub_per_day, 3), roi_per_hr=round(roi_per_hr, 5),
        score=round(sub_per_hr, 5),
    )


# --- driver -----------------------------------------------------------------
def screen(client: KalshiClient, *, capital: float, scan: int, top: int,
           deep: bool, exclude_toxic: bool, min_hours_left: float,
           min_price: float, max_price: float, pause: float) -> list[Candidate]:
    programs = fetch_active_programs(client)
    # Pre-rank cheaply by pool_per_hr (needs no market call) and only enrich the
    # top `scan` — bounds the per-market API calls well under the rate limit.
    for p in programs:
        s, e = _iso(p["start_date"]), _iso(p["end_date"])
        ph = max((e - s).total_seconds() / 3600, 1e-6)
        p["_rate"] = (_num(p.get("period_reward")) / REWARD_PER_DOLLAR) / ph
    programs.sort(key=lambda p: p["_rate"], reverse=True)

    seen: set[str] = set()
    cands: list[Candidate] = []
    for p in programs:
        if len(cands) >= scan:
            break
        tk = p["market_ticker"]
        if tk in seen:
            continue
        seen.add(tk)
        c = enrich(client, p, capital, deep, pause)
        if pause and not deep:
            time.sleep(pause)
        if not c:
            continue
        if c.hours_left < min_hours_left:
            continue
        if not (min_price <= c.price <= max_price):   # extreme longshots: skip
            continue
        if exclude_toxic and c.toxicity == "LIVE":
            continue   # by default we KEEP LIVE (flagged) — TOM proves a thin
                       # side is farmable if we exit before the catalyst; the
                       # 0.10 toxicity multiplier already haircuts its score.
        cands.append(c)

    # Collapse correlated strikes: many rows are adjacent strikes on ONE number
    # (5 gas-price thresholds = 1 bet). They reprice together, so farming several
    # stacks correlated adverse selection, not independent pools. Keep the single
    # best-scoring strike per event_ticker for the headline ranking.
    best_by_event: dict[str, Candidate] = {}
    for c in cands:
        key = c.event or c.ticker
        if key not in best_by_event or c.score > best_by_event[key].score:
            best_by_event[key] = c
    deduped = sorted(best_by_event.values(), key=lambda c: c.score, reverse=True)
    return deduped[:top]


def _print_table(cands: list[Candidate]) -> None:
    hdr = (f"{'#':>2} {'ticker':<32} {'$/hr':>6} {'pool':>6} {'toCl':>5} "
           f"{'px':>4} {'bidSd':>6} {'askSd':>6} {'thin':>6} {'lop':>5} "
           f"{'share':>6} {'tox':>6} {'sub$/hr':>7} {'sub$/day':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, c in enumerate(cands, 1):
        print(f"{i:>2} {c.ticker:<32.32} {c.pool_per_hr:>6.2f} {c.pool:>6.0f} "
              f"{c.hours_to_close:>5.0f} {c.price:>4.2f} {c.bid_side:>6.0f} "
              f"{c.ask_side:>6.0f} {c.thin_side:>6.0f} {c.lopsided:>5.1f} "
              f"{c.our_share:>6.1%} {c.toxicity:>6} {c.sub_per_hr:>7.3f} {c.sub_per_day:>8.2f}")
    if cands:
        print("\ntitles:")
        for i, c in enumerate(cands, 1):
            print(f"  {i:>2}. {c.title}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital-per-market", type=float, default=DEFAULT_CAPITAL_PER_MARKET,
                    help="$ reserved per market (≈ contracts quoted per side)")
    ap.add_argument("--scan", type=int, default=250,
                    help="how many top-by-pool-rate programs to enrich (API budget)")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--deep", action="store_true",
                    help="measure per-side competition within a 3¢ band from the book (1 extra call/market)")
    ap.add_argument("--exclude-toxic", action="store_true",
                    help="hard-drop LIVE (in-event) markets; default keeps them, flagged")
    ap.add_argument("--min-hours-left", type=float, default=2.0)
    ap.add_argument("--min-price", type=float, default=0.08)
    ap.add_argument("--max-price", type=float, default=0.92)
    ap.add_argument("--pause", type=float, default=0.05, help="sleep between calls (rate limit)")
    ap.add_argument("--json", type=str, default="", help="also write shortlist to this path")
    args = ap.parse_args()

    client = KalshiClient.from_env(host=CANDIDATE_HOSTS[0])
    cands = screen(
        client, capital=args.capital_per_market, scan=args.scan, top=args.top,
        deep=args.deep, exclude_toxic=args.exclude_toxic,
        min_hours_left=args.min_hours_left, min_price=args.min_price,
        max_price=args.max_price, pause=args.pause,
    )
    _print_table(cands)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(c) for c in cands], f, indent=1)
        print(f"\nwrote {len(cands)} → {args.json}")


if __name__ == "__main__":
    main()
