#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --replicator-time-span 20000 --replicator-time-steps 200000 --payoff-mode population-integral --output-dir BasicRun
