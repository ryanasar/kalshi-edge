"""
src/features/pitcher_form.py

Recent-form feature vector for a starting pitcher, computed from
`statcast_pitches`. Point-in-time correct: uses only data with
`game_date < as_of_date` — the model never sees a pitch that hadn't
been thrown yet at prediction time.

The features here are the pitcher-side inputs to the Layer-1 estimator
described in CLAUDE.md §5.5. They're all derived from Statcast physical
primitives + expected stats (§4) — no market-derived inputs.

Feature families produced:

    Volume     : games, pitches, batters_faced
    Rate       : k_pct, bb_pct, hr_per_9
    Stuff      : avg_velo (mph), avg_spin (rpm)
    Contact    : xwoba_against (Statcast expected wOBA on batted balls),
                 avg_exit_velo (mph)
    Platoon    : xwoba_vs_L, xwoba_vs_R, k_pct_vs_L, k_pct_vs_R
                 (see CLAUDE.md §4 — real ~15% swing effect)

Usage (one-off):
    ./.venv/bin/python -m src.features.pitcher_form \\
        --pitcher 665862 --as-of 2024-04-25 --lookback 30
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

from src.pipeline.db import connect


# --- SQL --------------------------------------------------------------------


# One aggregate over the pitcher's pitches in [as_of - lookback, as_of).
# The half-open window is deliberate: `game_date < as_of` excludes today's
# start (which we're trying to predict) even if today has data.
#
# Batted-ball metrics (xwOBA, avg_exit_velo) are conditional aggregates,
# not row filters, so they cost nothing extra to compute alongside rate
# stats.
BASE_SQL = """
SELECT
    COUNT(DISTINCT game_pk)                                     AS games,
    COUNT(*)                                                    AS pitches,
    COUNT(*) FILTER (WHERE events IS NOT NULL)                  AS pas,
    COUNT(*) FILTER (WHERE events = 'strikeout')                AS ks,
    COUNT(*) FILTER (WHERE events IN ('walk','hit_by_pitch'))   AS bbs,
    COUNT(*) FILTER (WHERE events = 'home_run')                 AS hrs,
    COUNT(*) FILTER (WHERE type = 'X')                          AS balls_in_play,

    -- Statcast expected wOBA on batted balls in play — the closest
    -- thing to a "true skill" estimate we have. Averaged over BIP only.
    AVG(estimated_woba_using_speedangle)
        FILTER (WHERE estimated_woba_using_speedangle IS NOT NULL) AS xwoba_bip,

    -- Stuff proxies. NULL on pitchouts, so include the FILTER.
    AVG(release_speed)     FILTER (WHERE release_speed     IS NOT NULL) AS avg_velo,
    AVG(release_spin_rate) FILTER (WHERE release_spin_rate IS NOT NULL) AS avg_spin,

    -- Batted-ball quality allowed. Only populated on BIP.
    AVG(launch_speed) FILTER (WHERE launch_speed IS NOT NULL) AS avg_exit_velo

FROM statcast_pitches
WHERE pitcher      = %(pitcher)s
  AND game_date   >= %(start)s
  AND game_date   <  %(as_of)s
  {stand_clause}
"""


@dataclass
class PitcherForm:
    pitcher_id: int
    as_of_date: date
    lookback_days: int

    # Volume
    games: int = 0
    pitches: int = 0
    pas: int = 0                     # plate appearances = end-of-PA rows
    balls_in_play: int = 0

    # Rates
    k_pct: float | None = None       # ks / pas
    bb_pct: float | None = None      # (bbs + hbps) / pas
    hr_per_9: float | None = None    # hrs * 9 / (pas * 0.25)  (crude IP proxy)

    # Stuff
    avg_velo: float | None = None
    avg_spin: float | None = None

    # Contact quality allowed
    xwoba_bip: float | None = None
    avg_exit_velo: float | None = None

    # Platoon splits (populated only if compute_platoon=True)
    xwoba_vs_L: float | None = None
    xwoba_vs_R: float | None = None
    k_pct_vs_L: float | None = None
    k_pct_vs_R: float | None = None
    pas_vs_L: int = 0
    pas_vs_R: int = 0


# --- Core aggregator --------------------------------------------------------


def _run_agg(conn, pitcher_id: int, start: date, as_of: date,
             stand: str | None) -> dict:
    stand_clause = "AND stand = %(stand)s" if stand else ""
    with conn.cursor() as cur:
        cur.execute(
            BASE_SQL.format(stand_clause=stand_clause),
            {"pitcher": pitcher_id, "start": start, "as_of": as_of,
             "stand": stand} if stand else
            {"pitcher": pitcher_id, "start": start, "as_of": as_of},
        )
        row = cur.fetchone()
    cols = [
        "games", "pitches", "pas", "ks", "bbs", "hrs", "balls_in_play",
        "xwoba_bip", "avg_velo", "avg_spin", "avg_exit_velo",
    ]
    return dict(zip(cols, row))


def _rate(numer: int | None, denom: int | None) -> float | None:
    """None-safe rate. Returns None when denom is 0 or None."""
    if not denom or not numer:
        return 0.0 if denom else None
    return numer / denom


def _to_float(v) -> float | None:
    """psycopg returns AVG() as Decimal; the rest of the pipeline uses
    plain floats. One-liner to normalize."""
    return float(v) if v is not None else None


def pitcher_recent_form(
    conn,
    pitcher_id: int,
    as_of_date: date,
    lookback_days: int = 30,
    compute_platoon: bool = True,
) -> PitcherForm:
    """
    Compute pitcher recent-form feature vector for the window
    [as_of_date - lookback_days, as_of_date). Never sees data at or after
    `as_of_date`.

    `compute_platoon=True` runs two extra aggregate queries — one for
    each batter stance — populating the four platoon-split fields.
    """
    start = as_of_date - timedelta(days=lookback_days)

    agg = _run_agg(conn, pitcher_id, start, as_of_date, stand=None)

    # Rough HR/9 proxy: pitchers throw ~4 batters per inning, so
    # PAs / 4 ≈ IP. Small-sample noisy but consistent with how the
    # sabermetric literature defines the ratio.
    ip_est = (agg["pas"] or 0) / 4.0
    hr_per_9 = (agg["hrs"] * 9.0 / ip_est) if ip_est > 0 else None

    form = PitcherForm(
        pitcher_id       = pitcher_id,
        as_of_date       = as_of_date,
        lookback_days    = lookback_days,
        games            = agg["games"] or 0,
        pitches          = agg["pitches"] or 0,
        pas              = agg["pas"] or 0,
        balls_in_play    = agg["balls_in_play"] or 0,
        k_pct            = _rate(agg["ks"], agg["pas"]),
        bb_pct           = _rate(agg["bbs"], agg["pas"]),
        hr_per_9         = hr_per_9,
        avg_velo         = _to_float(agg["avg_velo"]),
        avg_spin         = _to_float(agg["avg_spin"]),
        xwoba_bip        = _to_float(agg["xwoba_bip"]),
        avg_exit_velo    = _to_float(agg["avg_exit_velo"]),
    )

    if compute_platoon:
        for stand, k_pct_field, xwoba_field, pas_field in (
            ("L", "k_pct_vs_L", "xwoba_vs_L", "pas_vs_L"),
            ("R", "k_pct_vs_R", "xwoba_vs_R", "pas_vs_R"),
        ):
            sub = _run_agg(conn, pitcher_id, start, as_of_date, stand=stand)
            setattr(form, xwoba_field, _to_float(sub["xwoba_bip"]))
            setattr(form, k_pct_field, _rate(sub["ks"], sub["pas"]))
            setattr(form, pas_field, sub["pas"] or 0)

    return form


# --- CLI --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher", type=int, required=True,
                        help="MLB player ID (see MLB Stats API /people)")
    parser.add_argument("--as-of", required=True,
                        help="Compute features strictly before this date (YYYY-MM-DD)")
    parser.add_argument("--lookback", type=int, default=30,
                        help="Days of history to aggregate over (default 30)")
    parser.add_argument("--no-platoon", action="store_true",
                        help="Skip platoon-split aggregates (faster)")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)

    with connect() as conn:
        form = pitcher_recent_form(
            conn,
            pitcher_id      = args.pitcher,
            as_of_date      = as_of,
            lookback_days   = args.lookback,
            compute_platoon = not args.no_platoon,
        )

    print(f"\npitcher {form.pitcher_id}  as_of {form.as_of_date}  "
          f"(lookback {form.lookback_days}d)")
    print(f"  games:         {form.games}")
    print(f"  pitches:       {form.pitches}")
    print(f"  PAs:           {form.pas}")
    print(f"  balls in play: {form.balls_in_play}")
    print()

    def _fmt(v, spec):
        return f"{v:{spec}}" if v is not None else "     n/a"

    print("  --- rates ---")
    print(f"  K%:            {_fmt(form.k_pct, '6.3f')}")
    print(f"  BB%:           {_fmt(form.bb_pct, '6.3f')}")
    print(f"  HR/9 (proxy):  {_fmt(form.hr_per_9, '6.3f')}")
    print()
    print("  --- stuff ---")
    print(f"  avg velo:      {_fmt(form.avg_velo, '6.2f')} mph")
    print(f"  avg spin:      {_fmt(form.avg_spin, '6.0f')} rpm")
    print()
    print("  --- contact ---")
    print(f"  xwOBA (BIP):   {_fmt(form.xwoba_bip, '6.3f')}")
    print(f"  avg exit velo: {_fmt(form.avg_exit_velo, '6.2f')} mph")
    print()
    print("  --- platoon (BIP xwOBA / K%) ---")
    print(f"  vs L (n={form.pas_vs_L:>3}): xwOBA {_fmt(form.xwoba_vs_L, '6.3f')}  "
          f"K% {_fmt(form.k_pct_vs_L, '6.3f')}")
    print(f"  vs R (n={form.pas_vs_R:>3}): xwOBA {_fmt(form.xwoba_vs_R, '6.3f')}  "
          f"K% {_fmt(form.k_pct_vs_R, '6.3f')}")


if __name__ == "__main__":
    main()
