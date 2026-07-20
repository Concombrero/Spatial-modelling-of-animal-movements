#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --output-dir script/reproduce_day_night/GameTheory/output/pop_int/T1 --run-config script/reproduce_day_night/GameTheory/output/pop_int/T1/run_config.json --replicator-time-span 5000 --replicator-time-steps 50000
