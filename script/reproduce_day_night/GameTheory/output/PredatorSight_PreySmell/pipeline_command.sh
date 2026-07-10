#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --chi11 0.1 --chi22 0.1 --chi12 -0.35 --chi21 0.35 --number-of-cycles 4 --observation-window 2 --conversion-rate 0.15 --predation-rate 0.25 --predator-decay 0.04 --prey-growth 0.1 --prey-sight-radius 0.08 --prey-smell-radius 0.3 --predator-sight-radius 0.3 --predator-smell-radius 0.08 --weights 0.2 0.8 --replicator-time-span 20000 --replicator-time-steps 200000 --payoff-mode population-integral --output-dir script/reproduce_day_night/GameTheory/output/Story_1
