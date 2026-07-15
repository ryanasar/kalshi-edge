"""
src/features/game_features.py

Per-game feature assembler for the moneyline (home-win) model — CLAUDE.md
§5.5 Layer 1, v1 scope: starting-pitcher form + home-field only.

For every Final game this produces one row: metadata, the binary label
(`is_home_winner`), and the raw starting-pitcher feature vectors for both
teams. It is a *pure, point-in-time-correct extractor* and nothing more:

  • Point-in-time: features are computed with `as_of = official_date`, so
    `pitcher_recent_form`'s `game_date < as_of` cutoff guarantees the model
    never sees a pitch from the game it's predicting (or any later game).

  • Season-to-date form: the lookback window floors at March 1 of the
    game's year. There are no MLB regular-season games in Nov–Feb, so a
    March-1 floor is effectively opening day — the window captures *this
    season only* and never bleeds into the prior season's games.

Deliberately NOT done here (they belong in the model layer, src/model/):
  • Imputation of missing features. The league-average fill must be fit on
    the TRAIN split only; doing it here would leak test-set statistics into
    training. So a pitcher with no data yet (season debut, opening day)
    yields None for every feature, and the model's imputer fills it later.
  • The home−away differential. Same reason — it's computed after a
    leak-free imputation in the model layer.

All inputs are Statcast primitives (§4); no market-derived data enters.

Usage:
    # all Final games → CSV for the model layer to consume
    ./.venv/bin/python -m src.features.game_features --all \\
        --out outputs/features_moneyline.csv

    # a date range (inclusive), printed summary only
    ./.venv/bin/python -m src.features.game_features \\
        --start 2025-04-01 --end 2025-04-30
"""

from __future__ import annotations

import argparse
import csv
from datetime import date

from src.pipeline.db import connect
from src.features.pitcher_form import PitcherForm, pitcher_recent_form


# The v1 moneyline feature set — five starting-pitcher primitives (§5.5).
# Platoon splits and team-offense features are intentionally out of scope
# until the calibration curve shows they're needed (complexity discovered,
# not chosen — §5.5).
FEATURE_KEYS: list[str] = ["k_pct", "bb_pct", "hr_per_9", "xwoba_bip", "avg_velo"]

# Full output column order. Metadata + label first, then the raw per-side
# pitcher features, then sample-size context the model layer may turn into
# a low-confidence flag. `home_pp_pas` / `away_pp_pas` are the plate
# appearances backing each form vector — near-zero means "trust the
# imputed league average, not this pitcher's noisy season-to-date line."
OUTPUT_COLUMNS: list[str] = (
    ["game_pk", "official_date", "home_team_code", "away_team_code",
     "is_home_winner", "home_pp_id", "away_pp_id"]
    + [f"home_{k}" for k in FEATURE_KEYS]
    + [f"away_{k}" for k in FEATURE_KEYS]
    + ["home_pp_pas", "away_pp_pas"]
)


# --- Season-to-date lookback -------------------------------------------------


def _season_start(d: date) -> date:
    """Opening-day floor for the game's season. March 1 is safe: there are
    no regular-season games before late March, so nothing between it and
    real opening day exists to pull in, and it can never reach the prior
    season (which ends in October)."""
    return date(d.year, 3, 1)


def season_to_date_lookback(official_date: date) -> int:
    """Days from the season floor up to (but excluding) the game date.
    Feeds `pitcher_recent_form(lookback_days=...)`, whose window is
    [as_of - lookback, as_of) — i.e. this season's starts before today."""
    return (official_date - _season_start(official_date)).days


# --- Feature extraction ------------------------------------------------------


def _extract(form: PitcherForm | None) -> dict[str, float | None]:
    """Pull the v1 feature keys off a PitcherForm. None form (unknown or
    missing pitcher) → all-None, to be imputed downstream."""
    if form is None:
        return {k: None for k in FEATURE_KEYS}
    return {k: getattr(form, k) for k in FEATURE_KEYS}


def _form_for(conn, pitcher_id: int | None, as_of: date,
              lookback_days: int) -> PitcherForm | None:
    """Season-to-date form for one starter, or None if we have no pitcher
    id for that side (≈0.1% of Final games have a NULL probable pitcher)."""
    if pitcher_id is None:
        return None
    return pitcher_recent_form(
        conn, pitcher_id, as_of_date=as_of,
        lookback_days=lookback_days, compute_platoon=False,
    )


def assemble_game(conn, game: dict, lookback_days: int | None = None) -> dict:
    """
    Build the feature row for a single game dict (keys per GAMES_SQL below).

    `lookback_days=None` (default) means season-to-date. Passing an int
    overrides it with a fixed trailing window — useful only for
    experiments comparing form definitions.
    """
    official_date: date = game["official_date"]
    lb = season_to_date_lookback(official_date) if lookback_days is None else lookback_days

    home_form = _form_for(conn, game["home_probable_pitcher_id"], official_date, lb)
    away_form = _form_for(conn, game["away_probable_pitcher_id"], official_date, lb)

    row: dict = {
        "game_pk":        game["game_pk"],
        "official_date":  official_date.isoformat(),
        "home_team_code": game["home_team_code"],
        "away_team_code": game["away_team_code"],
        "is_home_winner": game["is_home_winner"],
        "home_pp_id":     game["home_probable_pitcher_id"],
        "away_pp_id":     game["away_probable_pitcher_id"],
        "home_pp_pas":    home_form.pas if home_form else 0,
        "away_pp_pas":    away_form.pas if away_form else 0,
    }
    for side, form in (("home", home_form), ("away", away_form)):
        for k, v in _extract(form).items():
            row[f"{side}_{k}"] = v
    return row


# --- Driver ------------------------------------------------------------------


GAMES_SQL = """
SELECT game_pk, official_date, home_team_code, away_team_code,
       home_probable_pitcher_id, away_probable_pitcher_id, is_home_winner
FROM games
WHERE status = 'Final'
  AND (%(start)s::date IS NULL OR official_date >= %(start)s)
  AND (%(end)s::date   IS NULL OR official_date <= %(end)s)
ORDER BY official_date, game_pk
"""


def _fetch_games(conn, start: date | None, end: date | None) -> list[dict]:
    cols = ["game_pk", "official_date", "home_team_code", "away_team_code",
            "home_probable_pitcher_id", "away_probable_pitcher_id",
            "is_home_winner"]
    with conn.cursor() as cur:
        cur.execute(GAMES_SQL, {"start": start, "end": end})
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def assemble(conn, start: date | None = None, end: date | None = None,
             lookback_days: int | None = None,
             verbose: bool = True) -> list[dict]:
    """Assemble feature rows for every Final game in [start, end]. Costs two
    index-backed aggregate queries per game (home + away starter)."""
    games = _fetch_games(conn, start, end)
    rows = [assemble_game(conn, g, lookback_days) for g in games]
    if verbose:
        _print_summary(rows)
    return rows


def _print_summary(rows: list[dict]) -> None:
    """Row count, label balance, and per-feature missingness — the numbers
    that tell you whether the imputation in the model layer will be doing a
    little work or a lot."""
    n = len(rows)
    print(f"\nassembled {n} game rows")
    if not n:
        return

    labeled = [r for r in rows if r["is_home_winner"] is not None]
    home_wins = sum(1 for r in labeled if r["is_home_winner"])
    print(f"  labeled:   {len(labeled)}  "
          f"(home wins {home_wins} = {home_wins / len(labeled):.3f})")

    print("  missing rate per feature (both sides):")
    for k in FEATURE_KEYS:
        miss = sum(1 for r in rows for side in ("home", "away")
                   if r[f"{side}_{k}"] is None)
        print(f"    {k:<12} {miss / (2 * n):6.3f}")


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows)} rows → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Assemble every Final game (ignores --start/--end)")
    parser.add_argument("--start", help="Start date (inclusive), YYYY-MM-DD")
    parser.add_argument("--end",   help="End date (inclusive), YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Fixed trailing-window days (default: season-to-date)")
    parser.add_argument("--out", help="Write assembled rows to this CSV path")
    args = parser.parse_args()

    if not args.all and not (args.start and args.end):
        parser.error("Provide --all, or --start AND --end")

    start = None if args.all else date.fromisoformat(args.start)
    end   = None if args.all else date.fromisoformat(args.end)

    with connect() as conn:
        rows = assemble(conn, start=start, end=end, lookback_days=args.lookback)

    if args.out:
        _write_csv(rows, args.out)


if __name__ == "__main__":
    main()
