#!/usr/bin/env bash
# scripts/plot_calibration.sh
#
# Runs the calibration analysis, printing the summary table + Brier score
# and saving the reliability diagram to outputs/calibration_market.png.
#
# Optional args are forwarded to the Python script:
#     bash scripts/plot_calibration.sh
#     bash scripts/plot_calibration.sh --bins 20
#     bash scripts/plot_calibration.sh --max-spread 0.20

set -euo pipefail

cd "$(dirname "$0")/.."

./.venv/bin/python -m src.analysis.calibration "$@"
