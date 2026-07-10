#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_payoff_pipeline.sh --output-dir ./script/reproduce_day_night/output/classicRun --replicator-time-span 2000
