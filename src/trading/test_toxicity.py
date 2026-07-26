"""Regression cases for the LIP toxicity classifier.

The classifier is the ONLY thing standing between the allocator and a toxic
market: `allocate.py` ranks by net EV, so anything mis-tiered as farmable gets
real money pointed at it. Every §5f loss so far traces to a gap here, and the
gaps are self-revealing in a nasty way — when one family gets demoted, the
allocator's top slots are immediately taken over by whatever is *still*
mis-tiered (demoting the sports next-team markets on 2026-07-25 promoted the
KXRAIN* rainfall markets straight into the top 5).

So this file pins BOTH directions:
  - TOXIC cases must stay non-farmable (regression guard on hard-won losses)
  - FARMABLE cases must stay farmable (guard against over-tightening the rules
    until the pipeline recommends nothing — every entry here has either a
    confirmed positive payout or is the same shape as one)

No pytest in this venv, so this runs standalone:

    PYTHONPATH="$PWD" .venv/bin/python -m src.trading.test_toxicity
"""

from __future__ import annotations

from src.trading.incentive_screen import classify_toxicity

# (ticker, title, expected_tier, why)
CASES: list[tuple[str, str, str, str]] = [
    # ---------------- TOXIC: unscheduled announcements (§5f generalized) -----
    # These look ideal — flat for days, tight book, thin side — but the
    # announcement lands at an UNKNOWN time so we can't pull the quote first.
    ("KXNEXTTEAMNBA-26KIRVING11-DAL", "What will be Kyrie Irving's next team?",
     "EVENT", "unscheduled trade/signing announcement"),
    ("KXJOINCLUB-26OCT02VINI7-MUN", "Where will Vinicius Junior go next?",
     "EVENT", "unscheduled transfer"),
    ("KXJOINLEAGUE-26OCT02MSALAH-MLS", "Where will Mohamed Salah go next?",
     "EVENT", "unscheduled transfer"),
    ("KXNEXTMANAGERMLB-BOS26-CTRA", "Who will be the next Manager of the Boston Red Sox?",
     "EVENT", "unscheduled hire"),
    ("KXINTLPLAYAGAIN-PORCRIS7-Y", "Will Cristiano Ronaldo ever play an official match",
     "EVENT", "unscheduled / open-ended"),
    ("KXMEDIACOVERSI-27-IRI", "Will Irina Shayk be on the cover of 2027 Sports Illustrated",
     "EVENT", "unscheduled reveal"),
    ("KXVOGUECOVER-27-TAY", "Will Taylor Swift be on the cover of any issue of Vogue (US)",
     "EVENT", "unscheduled reveal"),
    ("KXCLARITYVOTE-26JUL-AUG08", "Will the Senate vote on CLARITY Act?",
     "EVENT", "unscheduled legislative action"),

    # ---------------- TOXIC: live sensor / accumulating weather --------------
    ("KXRAINDENM-26JUL-2", "Will it rain more than 2 inches in Denver in July",
     "LIVE", "live sensor AND accumulating monthly total"),
    ("KXRAIN-26JUL27-SFO", "Will it rain in San Francisco on Jul 27",
     "LIVE", "live sensor"),
    ("KXAQICITY-26JUL-100", "Will the AQI be above 100", "LIVE", "live sensor"),
    ("KXTEMP-26JUL27-95", "Will the temperature in Austin be above 95",
     "LIVE", "live sensor"),

    # ---------------- TOXIC: spot-referenced (§5f drift) --------------------
    # NET WORTH tracked TSLA tick-by-tick; cost a realized -$5.70 on
    # KXMUSKNW-26JUL31-T750 and was one of the pins we crossed to flatten.
    ("KXMUSKNW-26JUL31-T750", "Will Elon Musk's net worth for July be above $750 billion?",
     "EVENT", "tracks TSLA continuously; measured -$5.70"),
    ("KXAAAGASM-26JUL31-4.12", "Will average gas prices be above $4.12?",
     "EVENT", "continuously-updating spot average, not a scheduled print"),
    # GPU pricing: only *MS (monthly average) is farmable. WS was killswitched
    # at -40; MON/MAX are spot-on-a-date / one-way ratchets.
    ("KXA100WS-26AUG07-1.000", "Will the A100 compute per hour price be above $1.00",
     "EVENT", "weekly spot; killswitched at -40 in run-2"),
    ("KXH200WS-26AUG21-6.000", "Will the H200 compute per hour price be above $6.00",
     "EVENT", "weekly spot"),
    ("KXB200MON-26JUL31-5.860", "Will the B200 compute per hour price be above $5.86 on Jul 31",
     "EVENT", "spot on a date"),
    ("KXH100MAX-26DEC31-3.380", "Will the H100 SXM compute per hour price be above $3.38 by Dec 31",
     "EVENT", "running max = one-way ratchet"),

    # ---------------- TOXIC: accumulating counts (§5f) ----------------------
    ("KXUSFLYCAN-26JUL31-T5500", "flight cancellations this week",
     "EVENT", "accumulating count; cost ~$22 in run-2"),
    ("KXEOWEEK-26AUG01-2", "Will the President sign more than 2 Executive Orders",
     "EVENT", "accumulating count"),
    ("KXBTCVSGOLD-26", "Will Bitcoin outperform gold", "EVENT", "crypto drift"),

    # ---------------- FARMABLE: scheduled official data prints --------------
    # All five run-1 payers plus the same-family names. These are the ones that
    # actually produced confirmed positive subsidy.
    ("KXUSGASCPI-26AUG12-T331", "Will Gasoline (All Types) in U.S. City Average for July 2026",
     "STABLE", "BLS scheduled release (NOT the AAA spot average)"),
    ("KXCPINDEX-26AUG12-T333.6", "Will CPI index be above", "STABLE", "run-1: +$8.35"),
    ("KXUSEDCARCPI-26AUG12-T179.25", "Will used car CPI be above", "STABLE", "run-1: +$6.74"),
    ("KXUSPPIYOY-26AUG13-T5.6", "Will PPI year over year be above", "STABLE", "run-1: +$5.57"),
    ("KXBUILDPERMS-26AUG18-T1.500", "Will building permits be above",
     "STABLE", "run-1: +$7.81, best take at 2.60%"),
    ("KXUSRETAIL-26AUG14-T1.0", "Will retail sales be above", "STABLE", "run-1: +$3.81"),
    ("KXNHSALES-26JUL24-T1", "Will new home sales be above", "STABLE", "run-2: +$1.08"),

    # ---------------- FARMABLE: GPU MONTHLY average (far-dated) -------------
    # Proven: 0 fills, 100% at-best, $2.89-$4.24 per $250 pool.
    ("KXA100MS-26OCT-1.250", "Will the monthly average compute price of NVIDIA's A100 be above",
     "MILD", "monthly average; proven 0-fill payer"),
    ("KXA100MS-27JUL-1.250", "Will the monthly average compute price of NVIDIA's A100 be above",
     "MILD", "monthly average"),
    ("KXRTX5090MS-26NOV-0.500", "Will the monthly average compute price of NVIDIA's RTX 5090 be above",
     "MILD", "run-2: +$3.43"),
    ("KXRTX5090MS-26SEP-0.750", "Will the monthly average compute price of NVIDIA's RTX 5090 be above",
     "MILD", "monthly average"),

    # ---------------- FARMABLE: corporate scheduled reports -----------------
    ("KXCMG-26JULREST-4170", "Will Chipotle Mexican Grill Inc. report Above 4170 total restaurants",
     "MILD", "reported at earnings = one scheduled release"),
    ("KXRDDT-26JULDAU-P129000000", "Will Reddit report above 129 million daily active uniques",
     "MILD", "reported at earnings"),
    ("KXNCLH-26JULPAX-840000", "Will Norwegian Cruise Line Holdings Ltd. report Above 840 thousand",
     "MILD", "reported at earnings"),
    ("KXHOODA-28JANFUNDED-31000000", "Will Robinhood Markets Inc. report above 31 million funded customers",
     "MILD", "reported at earnings"),
    ("KXAMZN-26OCTHEAD-1575000", "Will Amazon.com, Inc. report above 1575000 Employees",
     "MILD", "reported at earnings"),
    ("KXMETA-26SEPHEAD-76000", "Will Meta Platforms, Inc. report Above 76000 Headcount in Q2",
     "MILD", "reported at earnings"),

    # ---------------- FARMABLE: scheduled food/menu prices ------------------
    ("KXTBCRUNCHWRAP-26AUG02-T6.69", "Will the Crunchwrap price be above",
     "MILD", "run-2: +$3.19"),
    ("KXAMSAVO-26JUL24", "Will the avocado price be above",
     "MILD", "run-2: +$4.98, zero fills — best clean payout"),
    ("KXCFACHICKSAND-26AUG02-T5.42", "Will the chicken sandwich price be above",
     "MILD", "run-2: +$1.39"),
]

# Tiers we will actually point money at. Anything above this is haircut hard
# enough by tox_mult that allocate.py effectively skips it.
FARMABLE = {"STABLE", "MILD"}


def main() -> int:
    failures: list[str] = []
    for ticker, title, expected, why in CASES:
        got = classify_toxicity(ticker, title)
        if got != expected:
            # Flag the boundary crossing louder — that one is a money bug, an
            # exact-tier drift is only a calibration change.
            crossed = (got in FARMABLE) != (expected in FARMABLE)
            failures.append(
                f"  {'[FARMABLE BOUNDARY CROSSED] ' if crossed else ''}{ticker}\n"
                f"      got={got} expected={expected}  ({why})"
            )

    total = len(CASES)
    toxic = sum(1 for c in CASES if c[2] not in FARMABLE)
    if failures:
        print(f"FAILED {len(failures)}/{total} toxicity cases:\n" + "\n".join(failures))
        return 1
    print(f"ok — {total} toxicity cases pass "
          f"({toxic} must stay toxic, {total - toxic} must stay farmable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
