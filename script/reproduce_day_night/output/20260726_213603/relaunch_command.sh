#!/usr/bin/env bash
set -euo pipefail

script/reproduce_day_night/run_all_figures.sh --python .venv/bin/python --skip-weight-sweep --output-root script/reproduce_day_night/output/20260726_213603
