#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/output/Test2 --mean-x-axes cycle1\,cycle2 --replicator-time-span 2000 --replicator-time-steps 400000 --weights 0.3 1 --prey-sight-radius 0.06 --predator-sight-radius 0.18 --prey-smell-radius 0.18 --predator-smell-radius 0.06 --chi22 0 --chi21 0.35 --chi12 -0.35
