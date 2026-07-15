-- 003_baseball_universe.sql
--
-- The baseball side of the pipeline. Kept in a separate "universe" from
-- the Kalshi markets side (§5 / §5.5) — different grain, different
-- refresh cadence, joined only at analysis time via `market_game_link`.
--
-- Three tables:
--   games                  — one row per MLB game, outcomes + metadata
--   statcast_pitches       — pitch-level physics/expected stats from Savant
--   market_game_link       — Kalshi ticker → MLB game_pk mapping
--
-- Apply from repo root:
--   docker compose exec -T db psql -U kalshi -d kalshi_edge \
--     < db/migrations/003_baseball_universe.sql

BEGIN;

-- ---------------------------------------------------------------------------
-- games: one row per MLB game (regular season, playoffs, spring).
--
-- Populated by the /schedule?hydrate=linescore,probablePitcher,weather
-- ingester. UPSERT semantics — the same row is refreshed as a game's
-- status moves from "Scheduled" → "In Progress" → "Final", so outcome
-- columns become non-NULL only after finalization.
--
-- The precomputed outcome columns (`total_runs`, `f5_runs`, `rfi_runs`,
-- `went_to_extras`, ...) are what the calibration join actually reads.
-- Storing them as columns rather than deriving from the `innings` JSONB
-- on every query is a size-vs-speed tradeoff — pennies of storage buys
-- microseconds of read latency and an SQL-native filter surface.
-- ---------------------------------------------------------------------------
CREATE TABLE games (
    -- MLB Stats API's canonical game identifier. Stable across sources
    -- (Savant, Retrosheet, etc.) — the primary join key for the entire
    -- baseball-side data model.
    game_pk               INT         PRIMARY KEY,

    -- `official_date` is the date the game is officially assigned to,
    -- which for late-night finishes can differ from `gameDate`. This is
    -- the value that matches Kalshi ticker date fragments (§ticker-parse).
    official_date         DATE        NOT NULL,

    -- 'R' (regular), 'P' (playoff), 'S' (spring), 'E' (exhibition), etc.
    -- Model training excludes non-'R' by default — Statcast semantics
    -- and roster stability differ across game types.
    game_type             TEXT        NOT NULL,

    -- MLB Stats API state machine (Final / Scheduled / In Progress / ...).
    -- Outcome columns are only populated when status = 'Final'.
    status                TEXT        NOT NULL,

    -- Scheduled length. 9 for regulation, 7 for doubleheader minis
    -- (post-2020 rule). `went_to_extras` compares to this, not a hardcoded 9.
    scheduled_innings     INT         NOT NULL,

    -- --- Teams ---------------------------------------------------------
    -- Team IDs are MLB Stats API's canonical (stable across seasons).
    -- Team codes are the 2–4 char abbreviations Kalshi uses in tickers
    -- (KC, SF, CWS, ...) — persisted here so the ticker→game_pk lookup
    -- is one join on (official_date, home_team_code, away_team_code).
    home_team_id          INT         NOT NULL,
    home_team_code        TEXT        NOT NULL,
    home_team_name        TEXT        NOT NULL,
    away_team_id          INT         NOT NULL,
    away_team_code        TEXT        NOT NULL,
    away_team_name        TEXT        NOT NULL,

    -- --- Venue + context ----------------------------------------------
    venue_id              INT         NOT NULL,
    venue_name            TEXT        NOT NULL,

    -- Probable pitchers. NULLable because MLB doesn't always publish them
    -- in advance (early-season schedule, bullpen games). NULL after game
    -- start is a data quality flag we can surface at ingest time.
    home_probable_pitcher_id INT,
    away_probable_pitcher_id INT,

    -- Weather snapshot from /schedule hydration. JSONB because MLB's
    -- schema drifts here (temp is sometimes 'F' sometimes int) and
    -- we don't yet need to index it — pull typed values downstream.
    weather               JSONB,

    -- --- Outcomes (NULL until status='Final') --------------------------
    -- Precomputed at ingest from `innings` for cheap calibration joins.
    -- These are the labels for every KXMLB* market type in the modeling
    -- subset — moneyline (is_home_winner), totals (total_runs, f*_runs),
    -- team totals (home_score / away_score), extras (went_to_extras).
    home_score            INT,
    away_score            INT,
    is_home_winner        BOOLEAN,
    total_runs            INT,        -- home_score + away_score
    rfi_runs              INT,        -- runs in the first inning, both teams
    f3_runs               INT,        -- first 3 innings, both teams
    f5_runs               INT,        -- first 5 innings, both teams
    f7_runs               INT,        -- first 7 innings, both teams
    went_to_extras        BOOLEAN,    -- last_completed_inning > scheduled_innings

    -- --- Escape hatches ------------------------------------------------
    -- Raw linescore innings array — needed if we ever want per-half-
    -- inning breakdowns (currently not modeled but cheap to keep).
    innings               JSONB,

    -- Full raw schedule entry for the game — audit trail, and future-
    -- proofs adding columns without a re-ingest.
    raw                   JSONB       NOT NULL,

    ingested_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chosen indexes match the actual join surface:
--   • Ticker-to-game join uses (official_date, home_team_code, away_team_code)
--   • Statcast join uses (game_pk) directly (that's the PK, no extra index)
--   • Calibration-by-date queries filter on official_date
--   • Team-schedule queries filter on team_code + date
CREATE INDEX games_official_date_idx
    ON games(official_date);
CREATE INDEX games_date_teams_idx
    ON games(official_date, home_team_code, away_team_code);
CREATE INDEX games_home_team_date_idx
    ON games(home_team_code, official_date);
CREATE INDEX games_away_team_date_idx
    ON games(away_team_code, official_date);


-- ---------------------------------------------------------------------------
-- statcast_pitches: pitch-level physical data from Baseball Savant.
--
-- The "primitive physics" data source (§4). Every column here is
-- observable at the ballpark by radar and camera; nothing derives from
-- prediction-market prices or sportsbook lines. Aggregated downstream
-- into pitcher/batter feature vectors.
--
-- Column subset from Savant's 117-column CSV: pitcher/batter identity,
-- handedness, count/inning context, pitch physics, batted-ball physics,
-- and the three "expected" stats (xBA/xwOBA/xSLG) — the load-bearing
-- feature source for team run-scoring estimation.
--
-- Deliberately deferred to a second pass: fielder IDs, umpire, fielding
-- alignment, delta-win-exp columns. Add via a follow-up migration if the
-- calibration curve indicates they'd help.
-- ---------------------------------------------------------------------------
CREATE TABLE statcast_pitches (
    -- Composite PK identifies a pitch uniquely across all games.
    -- (game_pk, at_bat_number, pitch_number) is stable across Savant
    -- re-exports; ordering within an at-bat starts at 1.
    game_pk               INT         NOT NULL,
    at_bat_number         INT         NOT NULL,
    pitch_number          INT         NOT NULL,

    -- Redundant with games.official_date once we join, but kept flat so
    -- pitcher-recent-form window queries don't need a games join.
    game_date             DATE        NOT NULL,

    -- Identity. Pitcher / batter IDs are MLB Stats API canonical.
    pitcher               INT         NOT NULL,
    batter                INT         NOT NULL,
    p_throws              TEXT        NOT NULL,   -- 'L' | 'R'
    stand                 TEXT        NOT NULL,   -- batter stance: 'L' | 'R'

    -- 3-letter Savant team codes. These do NOT match Kalshi codes 1:1
    -- (Savant uses 'CWS' where Kalshi also uses 'CWS'; but Savant 'KC'
    -- vs Kalshi 'KC' — usually match, but confirmable at analysis time).
    home_team             TEXT        NOT NULL,
    away_team             TEXT        NOT NULL,

    -- Game-state context — the "situation" of the pitch.
    inning                INT         NOT NULL,
    inning_topbot         TEXT        NOT NULL,   -- 'Top' | 'Bot'
    outs_when_up          INT         NOT NULL,
    balls                 INT,
    strikes               INT,

    -- --- Pitch physics (all from radar/camera) ------------------------
    -- pitch_type is Savant's classifier ('FF', 'SL', 'CH', 'CU', ...);
    -- see MLB Stats API pitch code reference. NULL on pitchouts.
    pitch_type            TEXT,
    release_speed         NUMERIC,    -- mph out of hand
    release_spin_rate     NUMERIC,    -- rpm
    spin_axis             NUMERIC,    -- degrees
    pfx_x                 NUMERIC,    -- horizontal movement, ft
    pfx_z                 NUMERIC,    -- vertical movement, ft
    plate_x               NUMERIC,    -- horiz location at plate, ft
    plate_z               NUMERIC,    -- vert location at plate, ft
    zone                  INT,        -- Savant zone 1..14

    -- --- Outcome of this pitch ----------------------------------------
    -- `type`: 'S'=strike, 'B'=ball, 'X'=in play. Coarse-grain outcome.
    -- `description`: fine-grain ('called_strike', 'foul_tip', ...).
    -- `events`: at-bat resolution; populated only on the final pitch of
    -- an at-bat ('single', 'strikeout', 'walk', 'field_out', ...).
    type                  TEXT,
    description           TEXT,
    events                TEXT,

    -- --- Batted ball physics (populated only on balls in play) --------
    launch_speed          NUMERIC,    -- exit velocity, mph
    launch_angle          NUMERIC,    -- degrees
    bb_type               TEXT,       -- 'fly_ball' | 'ground_ball' | ...
    hit_distance_sc       INT,        -- feet
    bat_speed             NUMERIC,    -- mph, 2023+ era
    swing_length          NUMERIC,    -- ft, 2023+ era

    -- --- Expected stats (the load-bearing physics-primitive layer) ----
    -- Derived by Savant from launch_speed + launch_angle only; NOT from
    -- any prediction-market signal. Aggregate across recent starts to
    -- estimate a pitcher's true run-suppression skill.
    estimated_ba_using_speedangle     NUMERIC,   -- xBA
    estimated_woba_using_speedangle   NUMERIC,   -- xwOBA
    estimated_slg_using_speedangle    NUMERIC,   -- xSLG

    ingested_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);

-- Indexes match the aggregation queries we'll actually run:
--   • Pitcher-recent-form: pitches by pitcher over a date window
--   • Batter-recent-form:  pitches faced by batter over a date window
--   • Game-level joins:    already served by the composite PK's leading col
CREATE INDEX statcast_pitches_pitcher_date_idx
    ON statcast_pitches(pitcher, game_date);
CREATE INDEX statcast_pitches_batter_date_idx
    ON statcast_pitches(batter, game_date);


-- ---------------------------------------------------------------------------
-- market_game_link: ticker → game_pk mapping.
--
-- Materialized so calibration queries can join markets to outcomes with
-- one hop (no ticker-parsing at query time). Populated by the linker
-- (src/pipeline/link_markets.py), which reads unlinked markets, parses
-- the ticker's date+team fragment, and looks up game_pk in `games`.
--
-- `match_method` records how the link was made so we can audit later —
-- 'ticker_parse' for the clean parse path, others for future fallbacks
-- (e.g., 'manual' for disambiguation of doubleheaders).
-- ---------------------------------------------------------------------------
CREATE TABLE market_game_link (
    ticker                TEXT        PRIMARY KEY REFERENCES markets(ticker),
    game_pk               INT         NOT NULL REFERENCES games(game_pk),

    match_method          TEXT        NOT NULL,   -- 'ticker_parse' | ...

    linked_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Reverse lookup: "give me every Kalshi market on this game."
CREATE INDEX market_game_link_game_pk_idx
    ON market_game_link(game_pk);

COMMIT;
