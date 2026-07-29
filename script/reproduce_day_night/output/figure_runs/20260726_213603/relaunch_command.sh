#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_all_figures.sh --python .venv/bin/python --game-theory-only --output-root script/reproduce_day_night/output/figure_runs/20260726_213603
