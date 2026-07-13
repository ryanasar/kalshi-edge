"""
src/pipeline/db.py

Owns the Postgres connection and exposes small typed helpers for the two
tables we ingest into (`markets`, `market_candles`).

Design notes:

- One `psycopg.connect()` per script run. Fine for batch ingestion. When we
  add the long-running WebSocket streamer (which needs concurrent
  connections), promote `connect()` to a `psycopg_pool.ConnectionPool`.
  That's the one extension point.

- `ON CONFLICT DO NOTHING` on both upserts. Settled markets are immutable;
  a candle for a given (ticker, period_end, period_minutes) never changes.
  Running any ingester twice is a no-op — safe to cron.

- `Jsonb(dict)` wraps a Python dict for the `raw` column. psycopg 3 doesn't
  auto-coerce dicts to JSONB (they'd be sent as text and rejected). One line
  of ceremony, forgotten every time.

- Timestamps: Kalshi returns ISO 8601 strings for the metadata timestamps
  (`open_time`, `close_time`, ...); psycopg parses them directly into
  TIMESTAMPTZ, so we pass the string through. Candle `end_period_ts` is
  Unix seconds (int) and gets converted to `datetime` at ingest.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg
from psycopg import Connection
from psycopg.types.json import Jsonb


def connect() -> Connection:
    """
    Open a Postgres connection from the DATABASE_URL env var.

    Use as a context manager:
        with connect() as conn:
            insert_market(conn, m)
            # commits on clean exit, rolls back on exception
    """
    # Load .env lazily (only when someone actually connects) so analysis
    # scripts don't need to know they should call load_dotenv themselves.
    # `override=False` means an already-set DATABASE_URL wins over what's
    # in .env, which is what you want for prod-style deployments.
    from dotenv import load_dotenv

    load_dotenv(override=False)

    dsn = os.environ["DATABASE_URL"]
    return psycopg.connect(dsn)


# --- markets -----------------------------------------------------------------


def insert_market(conn: Connection, market: dict) -> bool:
    """
    Upsert one market. Returns True if a row was actually inserted, False
    if it already existed (ON CONFLICT DO NOTHING → 0 rows affected).

    `market` is the raw Kalshi /markets response element — the whole dict
    goes into `raw` for audit, and we lift the ~15 fields we index into
    typed columns.
    """
    event_ticker = market["event_ticker"]
    # Series ticker is the first '-'-separated segment of the event ticker
    # (e.g. 'KXFED-26JUN' → 'KXFED'). Kalshi doesn't return it separately.
    series_ticker = event_ticker.split("-", 1)[0]

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO markets (
                ticker, event_ticker, series_ticker,
                market_type, strike_type, floor_strike,
                title, rules_primary,
                open_time, close_time, occurrence_datetime, settlement_ts,
                status, result, settlement_value,
                raw
            )
            VALUES (
                %(ticker)s, %(event_ticker)s, %(series_ticker)s,
                %(market_type)s, %(strike_type)s, %(floor_strike)s,
                %(title)s, %(rules_primary)s,
                %(open_time)s, %(close_time)s, %(occurrence_datetime)s, %(settlement_ts)s,
                %(status)s, %(result)s, %(settlement_value)s,
                %(raw)s
            )
            ON CONFLICT (ticker) DO NOTHING
            """,
            {
                "ticker": market["ticker"],
                "event_ticker": event_ticker,
                "series_ticker": series_ticker,
                "market_type": market["market_type"],
                "strike_type": market["strike_type"],
                "floor_strike": market.get("floor_strike"),
                "title": market["title"],
                "rules_primary": market["rules_primary"],
                "open_time": market["open_time"],
                "close_time": market["close_time"],
                "occurrence_datetime": market.get("occurrence_datetime"),
                "settlement_ts": market.get("settlement_ts"),
                "status": market["status"],
                "result": market.get("result"),
                # settlement_value_dollars is the canonical string per §8.
                # NUMERIC column accepts the string directly — no float cast.
                "settlement_value": market.get("settlement_value_dollars"),
                # Jsonb wraps the dict so psycopg sends it as JSONB, not text.
                "raw": Jsonb(market),
            },
        )
        return cur.rowcount > 0


# --- market_candles ----------------------------------------------------------


def insert_candles(
    conn: Connection,
    ticker: str,
    period_minutes: int,
    candles: list[dict],
) -> int:
    """
    Batch-upsert candles for one ticker + one resolution. Returns count of
    rows actually inserted (excludes ON CONFLICT skips).

    Each `candle` is a raw element from Kalshi's /candlesticks response:
        {
          "end_period_ts": 1781632800,
          "open_interest_fp": "28683.92",
          "price": { "open_dollars": "...", "close_dollars": "...", ... },
          "yes_bid": { "close_dollars": "...", ... },
          "yes_ask": { "close_dollars": "...", ... },
          "volume_fp": "1000.00"
        }
    """
    if not candles:
        return 0

    # Flatten Kalshi's nested `price` / `yes_bid` / `yes_ask` objects into
    # flat columns, and convert Unix-second `end_period_ts` into aware UTC
    # `datetime` for TIMESTAMPTZ storage.
    rows: list[tuple] = []
    for c in candles:
        end_ts = datetime.fromtimestamp(c["end_period_ts"], tz=timezone.utc)
        price = c.get("price") or {}
        yes_bid = c.get("yes_bid") or {}
        yes_ask = c.get("yes_ask") or {}
        rows.append(
            (
                ticker,
                end_ts,
                period_minutes,
                price.get("open_dollars"),
                price.get("high_dollars"),
                price.get("low_dollars"),
                price.get("close_dollars"),
                price.get("mean_dollars"),
                yes_bid.get("close_dollars"),
                yes_ask.get("close_dollars"),
                c.get("volume_fp"),
                c.get("open_interest_fp"),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO market_candles (
                ticker, end_period_ts, period_minutes,
                open_dollars, high_dollars, low_dollars, close_dollars, mean_dollars,
                yes_bid_close, yes_ask_close,
                volume, open_interest
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, end_period_ts, period_minutes) DO NOTHING
            """,
            rows,
        )
        # psycopg 3 reports executemany's rowcount as the total across all
        # parameter sets — reliable here.
        return cur.rowcount


# --- data_sources + observations (feature-data side) ------------------------


def upsert_data_source(
    conn: Connection,
    series_id: str,
    description: str,
    provider: str,
    unit: str,
    is_vintaged: bool,
    metadata: dict | None = None,
) -> bool:
    """
    Register (or leave unchanged) a series in `data_sources`. ON CONFLICT DO
    NOTHING so a manual override in this table isn't clobbered by a later
    auto-registration.

    Returns True if a new row was inserted, False if it already existed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO data_sources
                (series_id, description, provider, unit, is_vintaged, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (series_id) DO NOTHING
            """,
            (
                series_id,
                description,
                provider,
                unit,
                is_vintaged,
                Jsonb(metadata) if metadata is not None else None,
            ),
        )
        return cur.rowcount > 0


def insert_observations(
    conn: Connection,
    rows: list[tuple[str, datetime, datetime, str | float, dict | None]],
) -> int:
    """
    Batch upsert observations. `rows` is a list of tuples:
        (series_id, observation_ts, published_ts, value, metadata_dict_or_None)

    ON CONFLICT DO NOTHING on the composite PK
    (series_id, observation_ts, published_ts) — a vintage is uniquely
    identified by that triple, so re-ingesting the same window is a no-op.
    """
    if not rows:
        return 0

    # psycopg needs Jsonb() wrapping for the metadata dicts.
    wrapped = [
        (series_id, obs_ts, pub_ts, value, Jsonb(meta) if meta is not None else None)
        for (series_id, obs_ts, pub_ts, value, meta) in rows
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO observations
                (series_id, observation_ts, published_ts, value, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (series_id, observation_ts, published_ts) DO NOTHING
            """,
            wrapped,
        )
        return cur.rowcount


# --- games (MLB schedule + linescore) ----------------------------------------


def upsert_game(conn: Connection, game_row: dict) -> bool:
    """
    Upsert one MLB game. ON CONFLICT (game_pk) — UPDATE so a game's row
    reflects its latest known state (schedule → in-progress → final).

    Returns True if this call actually changed a row (INSERT or UPDATE),
    False if the incoming row was identical to what's already stored.

    `game_row` is expected to already have derived outcome columns
    (total_runs, f5_runs, etc.) computed by the ingester — this helper
    is a thin persistence layer, not a business-logic one.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO games (
                game_pk, official_date, game_type, status, scheduled_innings,
                home_team_id, home_team_code, home_team_name,
                away_team_id, away_team_code, away_team_name,
                venue_id, venue_name,
                home_probable_pitcher_id, away_probable_pitcher_id,
                weather,
                home_score, away_score, is_home_winner,
                total_runs, rfi_runs, f3_runs, f5_runs, f7_runs, went_to_extras,
                innings, raw
            )
            VALUES (
                %(game_pk)s, %(official_date)s, %(game_type)s, %(status)s, %(scheduled_innings)s,
                %(home_team_id)s, %(home_team_code)s, %(home_team_name)s,
                %(away_team_id)s, %(away_team_code)s, %(away_team_name)s,
                %(venue_id)s, %(venue_name)s,
                %(home_probable_pitcher_id)s, %(away_probable_pitcher_id)s,
                %(weather)s,
                %(home_score)s, %(away_score)s, %(is_home_winner)s,
                %(total_runs)s, %(rfi_runs)s, %(f3_runs)s, %(f5_runs)s, %(f7_runs)s, %(went_to_extras)s,
                %(innings)s, %(raw)s
            )
            ON CONFLICT (game_pk) DO UPDATE SET
                status                   = EXCLUDED.status,
                scheduled_innings        = EXCLUDED.scheduled_innings,
                home_probable_pitcher_id = EXCLUDED.home_probable_pitcher_id,
                away_probable_pitcher_id = EXCLUDED.away_probable_pitcher_id,
                weather                  = EXCLUDED.weather,
                home_score               = EXCLUDED.home_score,
                away_score               = EXCLUDED.away_score,
                is_home_winner           = EXCLUDED.is_home_winner,
                total_runs               = EXCLUDED.total_runs,
                rfi_runs                 = EXCLUDED.rfi_runs,
                f3_runs                  = EXCLUDED.f3_runs,
                f5_runs                  = EXCLUDED.f5_runs,
                f7_runs                  = EXCLUDED.f7_runs,
                went_to_extras           = EXCLUDED.went_to_extras,
                innings                  = EXCLUDED.innings,
                raw                      = EXCLUDED.raw,
                updated_at               = NOW()
            """,
            {
                **game_row,
                "weather": Jsonb(game_row["weather"]) if game_row.get("weather") is not None else None,
                "innings": Jsonb(game_row["innings"]) if game_row.get("innings") is not None else None,
                "raw": Jsonb(game_row["raw"]),
            },
        )
        return cur.rowcount > 0


# --- statcast_pitches --------------------------------------------------------


# Kept in one place so the ingester, tests, and any future re-hydration
# code all agree on the exact column set + order used by insert_pitches.
STATCAST_COLUMNS = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "pitcher",
    "batter",
    "p_throws",
    "stand",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "type",
    "description",
    "events",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "hit_distance_sc",
    "bat_speed",
    "swing_length",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
)


def insert_pitches(conn: Connection, rows: list[tuple]) -> int:
    """
    Batch-insert Statcast pitches. `rows` is a list of tuples matching
    STATCAST_COLUMNS in order. ON CONFLICT DO NOTHING on the composite
    PK — re-ingesting the same date is a no-op.

    Returns the count of rows actually inserted (excludes conflicts).
    """
    if not rows:
        return 0

    placeholders = ", ".join(["%s"] * len(STATCAST_COLUMNS))
    columns = ", ".join(STATCAST_COLUMNS)

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO statcast_pitches ({columns})
            VALUES ({placeholders})
            ON CONFLICT (game_pk, at_bat_number, pitch_number) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount


# --- market_game_link --------------------------------------------------------


def insert_market_game_link(
    conn: Connection,
    ticker: str,
    game_pk: int,
    match_method: str,
) -> bool:
    """
    Record a ticker → game_pk link. ON CONFLICT DO NOTHING — the first
    successful link wins; a later re-run of the linker is idempotent.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO market_game_link (ticker, game_pk, match_method)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker) DO NOTHING
            """,
            (ticker, game_pk, match_method),
        )
        return cur.rowcount > 0


def list_unlinked_markets(
    conn: Connection,
    series_filter: str | None = None,
) -> list[tuple[str, str]]:
    """
    Return [(ticker, event_ticker), ...] for markets that aren't yet in
    `market_game_link`. Optionally filter to a single series.

    Used by the linker to know what work remains — anti-join against the
    link table so re-runs skip already-linked markets.
    """
    with conn.cursor() as cur:
        if series_filter:
            cur.execute(
                """
                SELECT m.ticker, m.event_ticker
                FROM markets m
                LEFT JOIN market_game_link l ON l.ticker = m.ticker
                WHERE l.ticker IS NULL AND m.series_ticker = %s
                ORDER BY m.ticker
                """,
                (series_filter,),
            )
        else:
            cur.execute(
                """
                SELECT m.ticker, m.event_ticker
                FROM markets m
                LEFT JOIN market_game_link l ON l.ticker = m.ticker
                WHERE l.ticker IS NULL
                ORDER BY m.ticker
                """
            )
        return list(cur.fetchall())


def lookup_game_pk(
    conn: Connection,
    official_date,
    home_team_code: str,
    away_team_code: str,
) -> int | None:
    """
    Return the game_pk for (date, home, away) or None if no match.

    NOTE on doubleheaders: two games with the same (date, home, away)
    exist. This function returns the FIRST match found; the linker
    handles the doubleheader disambiguation separately using the ticker's
    `G1`/`G2` fragment if present.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT game_pk FROM games
            WHERE official_date = %s
              AND home_team_code = %s
              AND away_team_code = %s
            ORDER BY game_pk
            LIMIT 1
            """,
            (official_date, home_team_code, away_team_code),
        )
        row = cur.fetchone()
        return row[0] if row else None


# --- reads used by other ingesters -------------------------------------------


def get_market_bounds(
    conn: Connection, ticker: str
) -> tuple[datetime, datetime] | None:
    """
    Return (open_time, close_time) for a ticker, or None if the ticker
    isn't in the markets table yet. Used by the candle ingester to bound
    its query window.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT open_time, close_time FROM markets WHERE ticker = %s",
            (ticker,),
        )
        row = cur.fetchone()
        return row if row else None
