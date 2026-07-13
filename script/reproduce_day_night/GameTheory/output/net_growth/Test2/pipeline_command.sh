#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/GameTheory/output/net_growth/Test2 --replicator-time-span 20000 --replicator-time-steps 400000 --weights 0.2 0.8 --prey-sight-radius 0.08 --predator-sight-radius 0.3 --prey-smell-radius 0.3 --predator-smell-radius 0.08 --payoff-mode net-growth
