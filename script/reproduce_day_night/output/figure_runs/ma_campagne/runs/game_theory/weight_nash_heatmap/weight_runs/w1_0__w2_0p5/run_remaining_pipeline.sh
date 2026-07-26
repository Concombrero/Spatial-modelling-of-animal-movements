#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=/home/tim/Documents/Spatial-modelling-of-animal-movements
PIPELINE_SCRIPT=/home/tim/Documents/Spatial-modelling-of-animal-movements/script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh

cd -- "$REPO_ROOT"

# This reuses the saved case_payoffs.csv in this folder when the payoff
# configuration matches, so finished payoff cases are not recomputed.
bash "$PIPELINE_SCRIPT" \
  --output-dir "$OUTPUT_DIR" \
  --python .venv/bin/python \
  --payoff-mode population-integral \
  --t-sunset 0.5 \
  --weights 0 0.5 \
  --prey-sight-radius 0.1 \
  --predator-sight-radius 0.1 \
  --prey-smell-radius 0.1 \
  --predator-smell-radius 0.1 \
  --number-of-points 128 \
  --dt 0.1 \
  --number-of-cycles 4 \
  --observation-window 2 \
  --prey-growth 0.1 \
  --predator-decay 0.04 \
  --predation-rate 0.25 \
  --conversion-rate 0.15 \
  --chi11 0.1 \
  --chi12 -0.2 \
  --chi21 0.2 \
  --chi22 0.1 \
  --diffusion 0.04 0.04 \
  --initial-centers 0.25 0.7 \
  --initial-width 0.1 \
  --strategy-codes D,N,P1,P2,M1,M2 \
  --max-workers 4
