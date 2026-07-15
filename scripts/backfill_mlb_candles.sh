#!/usr/bin/env bash
# scripts/backfill_mlb_candles.sh
#
# Backfills full-life hourly candles for every ticker in the MLB modeling
# series. Idempotent (ON CONFLICT DO NOTHING).
#
# Run:
#     bash scripts/backfill_mlb_candles.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# Modeling-focus series. Player-prop series are in backfill_series.py's
# CANDIDATE_SERIES for archive purposes but not modeled in Week 2.
series_list=(
    KXMLBGAME
    KXMLBTOTAL
    KXMLBF5TOTAL
    KXMLBSPREAD
    KXMLBTEAMTOTAL
)

for s in "${series_list[@]}"; do
    echo ""
    echo "=== $s ==="
    ./.venv/bin/python -m src.pipeline.backfill_candles --series "$s"
done

echo ""
echo "Done. MLB modeling-series candles backfilled."
