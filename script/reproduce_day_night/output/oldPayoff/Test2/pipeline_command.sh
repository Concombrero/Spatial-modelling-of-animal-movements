#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/output/Test2 --python .venv/bin/python --mean-x-axes cycle1\,cycle2 --t-sunset 0.5 --weights 0.3 0.7 --number-of-points 128 --dt 0.1 --number-of-cycles 4 --observation-window 2.0 --replicator-time-span 12000 --replicator-time-steps 24000 --replicator-plot-style line --prey-sight-radius 0.08 --prey-smell-radius 0.35 --predator-smell-radius 0.08 --predator-sight-radius 0.35 --chi11 0 --chi12 -0.35 --chi21 0.35 --chi22 0
