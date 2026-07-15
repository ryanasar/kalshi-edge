-- 001_init_markets.sql
--
-- Kalshi-side tables: the markets we're pricing against.
--
-- `markets` holds one immutable row per market ticker (created when we
-- ingest a resolved market from Kalshi's /markets endpoint). `market_candles`
-- holds the historical price time-series for each ticker, one row per
-- (ticker, period-end, period-length) triple.
--
-- Apply from repo root:
--   docker compose exec -T db psql -U kalshi -d kalshi_edge \
--     < db/migrations/001_init_markets.sql

BEGIN;

-- ---------------------------------------------------------------------------
-- markets: one row per Kalshi market.
--
-- Populated by our REST ingestion job from /trade-api/v2/markets. Rows are
-- effectively immutable after `status = 'finalized'`; for still-open markets
-- we may UPSERT snapshot-like fields as they evolve.
-- ---------------------------------------------------------------------------
CREATE TABLE markets (
    -- Identity. `ticker` is Kalshi's unique per-market ID (e.g.
    -- 'KXFED-26JUN-T4.25'). `event_ticker` groups all markets for the same
    -- underlying event; `series_ticker` groups all events for the same
    -- underlying question type (all Fed meetings, all CPI prints, etc.).
    -- Derived at ingest as event_ticker.split('-')[0] — stored explicitly
    -- so we can index it cheaply.
    ticker              TEXT PRIMARY KEY,
    event_ticker        TEXT        NOT NULL,
    series_ticker       TEXT        NOT NULL,

    -- Question shape. These four are what our probability engine reads to
    -- know what threshold/direction each market settles on. `market_type`
    -- has been 'binary' in every sample so far; kept as TEXT for forward
    -- compat if Kalshi adds range/scalar markets. `floor_strike` is NUMERIC
    -- because §8 — never store price/threshold values as FLOAT.
    market_type         TEXT        NOT NULL,
    strike_type         TEXT        NOT NULL,   -- 'greater', 'less', 'between', ...
    floor_strike        NUMERIC,                -- NULL for non-strike markets

    -- Human-readable question + authoritative resolution rules. Redundant
    -- with the structured fields for programmatic use, but load-bearing for
    -- audit ("what did Kalshi actually promise this market would settle on?").
    title               TEXT        NOT NULL,
    rules_primary       TEXT        NOT NULL,

    -- Timing. Every timestamp is TIMESTAMPTZ so it survives DST and cross-
    -- timezone reads without lying. `close_time` is the "as-of" moment for
    -- calibration — the last instant we could have quoted a probability.
    open_time           TIMESTAMPTZ NOT NULL,
    close_time          TIMESTAMPTZ NOT NULL,
    occurrence_datetime TIMESTAMPTZ,            -- the underlying event's scheduled time
    settlement_ts       TIMESTAMPTZ,            -- when Kalshi actually settled it

    -- Resolution. `status` is Kalshi's state machine ('active', 'finalized'
    -- etc.); `result` is 'yes' | 'no' | NULL if still open. `settlement_value`
    -- is $1.00 for YES-won, $0.00 for NO-won, potentially fractional if we
    -- ever hit partial-payout markets.
    status              TEXT        NOT NULL,
    result              TEXT,
    settlement_value    NUMERIC,

    -- The escape hatch. `raw` is the full Kalshi response as JSONB — we
    -- extract the ~10 fields above into typed columns for querying, but
    -- keep the whole payload so:
    --   (a) audit questions ("did Kalshi actually send us this?") stay
    --       answerable;
    --   (b) adding a column later doesn't require re-ingesting.
    -- JSONB compresses well; the redundancy with extracted columns is
    -- negligible at our row count.
    raw                 JSONB       NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes. Chosen to match how the calibration study, live dashboard, and
-- backtest queries will actually filter — not speculative.
CREATE INDEX markets_event_ticker_idx  ON markets(event_ticker);
CREATE INDEX markets_series_ticker_idx ON markets(series_ticker);
CREATE INDEX markets_close_time_idx    ON markets(close_time);
CREATE INDEX markets_status_idx        ON markets(status);


-- ---------------------------------------------------------------------------
-- market_candles: one row per (ticker, period-end, period-length).
--
-- Populated by our REST ingestion job from
-- /trade-api/v2/series/{series}/markets/{ticker}/candlesticks and, later,
-- filled forward by the WebSocket ingester for still-open markets.
--
-- Note: Kalshi's candlesticks endpoint SKIPS periods with no activity, so
-- the timestamp axis here is sparse — never assume "one row per hour."
-- ---------------------------------------------------------------------------
CREATE TABLE market_candles (
    -- FK to markets ensures we never orphan candles for a ticker we don't
    -- know about. Cascading isn't needed because we don't delete markets.
    ticker              TEXT        NOT NULL REFERENCES markets(ticker),

    -- End of this candle's period. Kalshi returns Unix seconds; we convert
    -- to TIMESTAMPTZ at ingest so SQL date math works directly.
    end_period_ts       TIMESTAMPTZ NOT NULL,

    -- Length of this candle in minutes. In the PK so we can hold multiple
    -- resolutions for the same ticker without collision (e.g. daily +
    -- hourly for a long-lived market).
    period_minutes      INT         NOT NULL,

    -- Last-trade OHLC + mean, from Kalshi's `price` object. All NUMERIC.
    -- `mean_dollars` is Kalshi's volume-weighted average trade price in the
    -- period; useful when you want a single "typical trade" number and the
    -- period had multiple prints.
    open_dollars        NUMERIC,
    high_dollars        NUMERIC,
    low_dollars         NUMERIC,
    close_dollars       NUMERIC,
    mean_dollars        NUMERIC,

    -- Best YES bid/ask at end of period, from Kalshi's `yes_bid` / `yes_ask`
    -- objects. Mid = (yes_bid_close + yes_ask_close) / 2 is what we compare
    -- against our model's implied probability. Bid + ask separately are used
    -- for honest fills in the backtest — sell YES at bid, buy YES at ask.
    yes_bid_close       NUMERIC,
    yes_ask_close       NUMERIC,

    -- Activity. Kalshi's `_fp` suffix is a string-encoded decimal, not
    -- fixed-point integer; store as NUMERIC to preserve exact value.
    volume              NUMERIC,
    open_interest       NUMERIC,

    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (ticker, end_period_ts, period_minutes)
);

-- Workhorse index for backtest queries: "give me the most recent candle for
-- this ticker at or before some timestamp." DESC so ORDER BY ... LIMIT 1 is
-- a single b-tree walk.
CREATE INDEX market_candles_ticker_end_idx
    ON market_candles(ticker, end_period_ts DESC);

COMMIT;
