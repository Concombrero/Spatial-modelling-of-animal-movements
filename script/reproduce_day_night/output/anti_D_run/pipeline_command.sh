#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/output/anti_D_run --python python --mean-x-axes cycle1\,cycle2 --weights 0.3 0.7 --prey-sight-radius 0.04 --predator-sight-radius 0.25 --predation-rate 0.22 --chi21 0.35 --chi12 0.20 --replicator-time-span 2000 --replicator-time-steps 2000
