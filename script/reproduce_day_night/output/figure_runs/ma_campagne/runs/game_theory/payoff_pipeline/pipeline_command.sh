#!/usr/bin/env bash
set -euo pipefail

/home/tim/Documents/Spatial-modelling-of-animal-movements/script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh --output-dir /home/tim/Documents/Spatial-modelling-of-animal-movements/script/reproduce_day_night/output/figure_runs/ma_campagne/runs/game_theory/payoff_pipeline --python venv/bin/python --payoff-mode population-integral
