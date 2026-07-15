-- 002_init_observations.sql
--
-- Feature-data tables: every external signal we use to build our own
-- probability estimate. Cleveland Fed nowcasts, ADP, WTI oil, PMI, rent
-- indices, etc. Also the destination for model-produced predictions.
--
-- Shape is deliberately different from the markets tables. `markets` is
-- optimized for Kalshi's specific heterogeneous API response (many typed
-- columns + JSONB backup). Feature data has the opposite shape: many
-- independent sources each producing simple (timestamp, value) observations,
-- so we use a single long-format table with source identity as a column.
--
-- The load-bearing design constraint is VINTAGING (see CLAUDE.md §8).
-- Every observation carries TWO timestamps:
--   observation_ts — the moment the observation is about (e.g. "March 2026 CPI")
--   published_ts   — the moment we could have first SEEN this value
-- The distinction is what makes lookahead-free backtests possible. Point-
-- in-time (PIT) database pattern; this is standard finance-infra vocabulary.
--
-- Apply:
--   docker compose exec -T db psql -U kalshi -d kalshi_edge \
--     < db/migrations/002_init_observations.sql

BEGIN;

-- ---------------------------------------------------------------------------
-- data_sources: one row per series we track. A lookup, not a fact table.
--
-- Rows are inserted manually (or via a small config file) when we wire up
-- a new feed — this is the "roster of things we know how to ingest."
-- ---------------------------------------------------------------------------
CREATE TABLE data_sources (
    -- Human-legible ID. Uppercase-snake convention makes them easy to grep
    -- for in code. Examples:
    --   'CLEVELAND_CPI_NOWCAST', 'CLEVELAND_PCE_NOWCAST',
    --   'ATLANTA_GDPNOW', 'ADP_TOTAL_PRIVATE',
    --   'WTI_FRONT_MONTH', 'ISM_MFG_PMI',
    --   'BLS_CPI_URBAN', ...
    series_id           TEXT PRIMARY KEY,

    -- One-line English description for docs/dashboards.
    description         TEXT NOT NULL,

    -- Where the data comes FROM operationally. 'cleveland_fed', 'alfred',
    -- 'eia', 'adp', 'polygon', etc. Not the same as `series_id` because
    -- one provider may host many series (ALFRED alone hosts thousands).
    provider            TEXT NOT NULL,

    -- Units the raw value is expressed in. Kept as freeform TEXT because
    -- a strict enum would fight us every time we add a new signal. Examples:
    -- 'percent', 'usd_per_barrel', 'jobs_thousands', 'index_2015=100'.
    unit                TEXT NOT NULL,

    -- Whether we track publish revisions for this series. TRUE for anything
    -- that gets revised (CPI, GDP, employment). FALSE for market prices
    -- (WTI oil doesn't get "revised" — the print is the print). Governs
    -- whether observations for this series can have published_ts >
    -- observation_ts (revised value) or must have them equal.
    is_vintaged         BOOLEAN NOT NULL,

    -- Freeform per-source config: ALFRED vintage series ID, upstream URL,
    -- release schedule notes, etc.
    metadata            JSONB
);


-- ---------------------------------------------------------------------------
-- observations: one row per (series, observation moment, publish moment).
--
-- Long format. Every external signal we ingest ends up here — one uniform
-- shape across all sources means adding a new signal is a data insert, not
-- a schema change. Model-produced predictions live here too (with a
-- synthetic series_id like 'MODEL_ENSEMBLE_CPI_v1').
-- ---------------------------------------------------------------------------
CREATE TABLE observations (
    -- FK to data_sources — every observation belongs to a known series.
    series_id           TEXT NOT NULL REFERENCES data_sources(series_id),

    -- WHAT period this observation is about. For CPI, this is the month
    -- being nowcast/measured. For WTI oil, it's the tick timestamp.
    observation_ts      TIMESTAMPTZ NOT NULL,

    -- WHEN we could first have seen this value. For non-vintaged series,
    -- ingest sets this equal to observation_ts. For vintaged series (Fed
    -- nowcasts, BLS prints), this is the actual publication time — and a
    -- single observation_ts may have MULTIPLE rows with different
    -- published_ts values as revisions land. THIS is what makes point-in-
    -- time backtests correct.
    published_ts        TIMESTAMPTZ NOT NULL,

    -- The value itself. NUMERIC per §8. If a source publishes multiple
    -- quantities per release (e.g., headline and core CPI), each gets its
    -- own series_id and its own row here.
    value               NUMERIC NOT NULL,

    -- Per-observation extras: the specific ALFRED vintage tag, error bars
    -- if the source publishes them, etc.
    metadata            JSONB,

    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- (series_id, observation_ts, published_ts) uniquely identifies a
    -- vintage of a single observation. Two publish dates for the same
    -- observation month = two rows. That IS the revision history.
    PRIMARY KEY (series_id, observation_ts, published_ts)
);

-- Workhorse index for the core backtest query: "give me the LATEST value
-- for series X that was published on or before backtest date T." DESC on
-- published_ts so ORDER BY ... LIMIT 1 walks the b-tree once.
CREATE INDEX observations_series_pub_idx
    ON observations(series_id, published_ts DESC);

-- Secondary index for the modeling query: "give me all versions of the
-- estimate for CPI month M, across every publish date." Useful for
-- studying revision behavior.
CREATE INDEX observations_series_obs_idx
    ON observations(series_id, observation_ts);

COMMIT;
