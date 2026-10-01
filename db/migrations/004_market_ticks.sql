-- ---------------------------------------------------------------------------
-- 004_market_ticks.sql
--
-- Raw live tick capture from the Kalshi WebSocket feed. This is the
-- forward-only microstructure dataset the hourly REST candles can't give
-- us — the substrate for market-making simulation (resting-order fills,
-- adverse selection) that pure OHLC bars cannot support.
--
-- We store every book frame (snapshot / delta) and trade verbatim in `raw`
-- (JSONB, lossless) plus the extracted columns we query on. Prices are
-- NUMERIC (the §8 fixed-point dollar strings — never FLOAT).
--
-- `recv_ts` is OUR receive wall-clock; `exch_ts_ms` is the exchange's. The
-- gap between them is measured feed latency — the number that decides
-- whether any event-reaction edge is even reachable.
-- ---------------------------------------------------------------------------
BEGIN;

CREATE TABLE market_ticks (
    id           BIGSERIAL   PRIMARY KEY,
    ticker       TEXT        NOT NULL,

    recv_ts      TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- when WE got it
    exch_ts_ms   BIGINT,                              -- exchange ts (ms)
    seq          BIGINT,                              -- per-subscription seq

    type         TEXT        NOT NULL,   -- orderbook_snapshot | orderbook_delta | trade | ticker
    side         TEXT,                   -- yes | no  (deltas / trades)
    price        NUMERIC,                -- dollars
    size         NUMERIC,                -- fixed-point size (delta for deltas)

    raw          JSONB       NOT NULL    -- the full frame, lossless
);

-- Primary access pattern: replay one market's ticks in time order.
CREATE INDEX market_ticks_ticker_recv_idx ON market_ticks (ticker, recv_ts);

COMMIT;
