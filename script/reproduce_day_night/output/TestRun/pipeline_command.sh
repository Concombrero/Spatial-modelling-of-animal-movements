#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir TestRun --mean-x-axes cycle1\,cycle2 --replicator-time-span 2000 --replicator-time-steps 400000 --weights 0.3 0.7 --prey-sight-radius 0.06 --predator-sight-radius 0.12 --prey-smell-radius 0.12 --predator-smell-radius 0.06
