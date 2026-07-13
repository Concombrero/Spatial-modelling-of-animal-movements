#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/GameTheory/output/net_growth/Test1 --replicator-time-span 20000 --replicator-time-steps 400000 --payoff-mode net-growth
