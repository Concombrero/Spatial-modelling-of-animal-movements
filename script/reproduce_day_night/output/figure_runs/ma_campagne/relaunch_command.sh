#!/usr/bin/env bash
set -euo pipefail

./script/reproduce_day_night/run_all_figures.sh --python venv/bin/python --output-root script/reproduce_day_night/output/figure_runs/ma_campagne
