"""
src/features/team_offense.py

Season-to-date team-offense feature vectors — the batting-side mirror of
src/features/pitcher_form.py, and the resolution lever for the moneyline
model (§5.5). A pitcher-only model matches the market's calibration but
can't separate games; team offense is what lets predictions spread.

Attribution — the one subtle part. A Statcast pitch belongs to the team
that's *batting*, which we read off `inning_topbot` against the game's
known codes (never Statcast's own abbreviations):

    inning_topbot = 'Top'  →  the AWAY team is batting
    inning_topbot = 'Bot'  →  the HOME team is batting

Performance — there's no (team, date) index on statcast_pitches, so rather
than run one date-range scan per (team, as_of) we do a SINGLE
GROUP BY (date, batting_team) pass (~12.5k rows, ~0.3s) and then sum a
team's prior games in Python. One scan instead of thousands.

Point-in-time: `as_of(team, d)` sums only games with date < d, and floors
at the season start, so a team's offense as-of game day reflects only what
had been played — no leakage, same discipline as pitcher_form.

Features produced (all Statcast primitives, offense perspective):

    off_k_pct     strikeout rate    (lower = better offense)
    off_bb_pct    walk rate         (higher = better plate discipline)
    off_xwoba     expected wOBA on batted balls (higher = better contact)
    off_exit_velo average exit velocity, mph    (higher = harder contact)

Usage (one-off inspection):
    ./.venv/bin/python -m src.features.team_offense --team NYY --as-of 2025-06-01
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from src.pipeline.db import connect

FEATURE_KEYS: list[str] = ["off_k_pct", "off_bb_pct", "off_xwoba", "off_exit_velo"]


# One row per (game date, batting team). The batting team is derived from
# inning_topbot against the game's home/away codes, so we depend on the
# games table (trusted, Kalshi-aligned) rather than Statcast abbreviations.
# xwOBA / exit velo are carried as (sum, n) so season-to-date averages can
# be reconstructed by summing across games — you can't average an average.
AGG_SQL = """
SELECT
    g.official_date AS gdate,
    CASE WHEN sp.inning_topbot = 'Top' THEN g.away_team_code
         ELSE g.home_team_code END                              AS bat_team,
    COUNT(*) FILTER (WHERE sp.events IS NOT NULL)               AS pas,
    COUNT(*) FILTER (WHERE sp.events = 'strikeout')             AS ks,
    COUNT(*) FILTER (WHERE sp.events IN ('walk','hit_by_pitch')) AS bbs,
    SUM(sp.estimated_woba_using_speedangle)
        FILTER (WHERE sp.estimated_woba_using_speedangle IS NOT NULL) AS xwoba_sum,
    COUNT(*) FILTER (WHERE sp.estimated_woba_using_speedangle IS NOT NULL) AS xwoba_n,
    SUM(sp.launch_speed) FILTER (WHERE sp.launch_speed IS NOT NULL) AS ev_sum,
    COUNT(*)            FILTER (WHERE sp.launch_speed IS NOT NULL) AS ev_n
FROM statcast_pitches sp
JOIN games g ON g.game_pk = sp.game_pk
WHERE sp.inning_topbot IN ('Top', 'Bot')
GROUP BY 1, 2
"""


@dataclass
class _GameAgg:
    """Raw batting components for one team on one date — summable across a
    season to reconstruct season-to-date rates."""
    pas: int
    ks: int
    bbs: int
    xwoba_sum: float
    xwoba_n: int
    ev_sum: float
    ev_n: int


def _rate(numer: int, denom: int) -> float | None:
    return (numer / denom) if denom else None


class TeamOffenseTable:
    """Precomputed per-(team, date) batting aggregates. Build once per run;
    then `as_of(team, d, season_start)` is a cheap in-memory sum."""

    def __init__(self, by_team: dict[str, list[tuple[date, _GameAgg]]]):
        self._by_team = by_team

    @classmethod
    def build(cls, conn) -> "TeamOffenseTable":
        by_team: dict[str, list[tuple[date, _GameAgg]]] = {}
        with conn.cursor() as cur:
            cur.execute(AGG_SQL)
            for (gdate, team, pas, ks, bbs,
                 xw_sum, xw_n, ev_sum, ev_n) in cur.fetchall():
                by_team.setdefault(team, []).append((
                    gdate,
                    _GameAgg(pas or 0, ks or 0, bbs or 0,
                             float(xw_sum or 0.0), xw_n or 0,
                             float(ev_sum or 0.0), ev_n or 0),
                ))
        # Sort each team's games by date so as_of can stop early.
        for games in by_team.values():
            games.sort(key=lambda t: t[0])
        return cls(by_team)

    def as_of(self, team: str, as_of_date: date,
              season_start: date) -> dict[str, float | None]:
        """
        Season-to-date offense for `team` over [season_start, as_of_date).
        Strictly before as_of_date, so the game being predicted (and any
        same-day games) never leak in. All-None if the team has no prior
        games this season yet.
        """
        pas = ks = bbs = xw_n = ev_n = 0
        xw_sum = ev_sum = 0.0
        for gdate, a in self._by_team.get(team, []):
            if gdate < season_start:
                continue
            if gdate >= as_of_date:
                break  # sorted by date — nothing later qualifies
            pas += a.pas
            ks += a.ks
            bbs += a.bbs
            xw_sum += a.xwoba_sum
            xw_n += a.xwoba_n
            ev_sum += a.ev_sum
            ev_n += a.ev_n

        return {
            "off_k_pct":     _rate(ks, pas),
            "off_bb_pct":    _rate(bbs, pas),
            "off_xwoba":     (xw_sum / xw_n) if xw_n else None,
            "off_exit_velo": (ev_sum / ev_n) if ev_n else None,
        }


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help="Team code (e.g. NYY)")
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD (exclusive)")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    season_start = date(as_of.year, 3, 1)

    with connect() as conn:
        table = TeamOffenseTable.build(conn)
        feats = table.as_of(args.team, as_of, season_start)

    print(f"\n{args.team} offense as-of {as_of} (season-to-date):")
    for k in FEATURE_KEYS:
        v = feats[k]
        print(f"  {k:<14} {v:.4f}" if v is not None else f"  {k:<14} n/a")


if __name__ == "__main__":
    main()
