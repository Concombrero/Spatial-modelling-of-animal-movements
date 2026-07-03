#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir classicRun --mean-x-axes cycle1\,cycle2 --replicator-time-span 2000 --replicator-time-steps 400000
