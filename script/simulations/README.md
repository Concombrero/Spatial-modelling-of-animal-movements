# Simulation Scripts

Runnable scripts for the day-night figures and the game-theory analyses. Default parameters come from `shared_config.py`. Run shell entry points with `bash` and Python entry points from the repository root with `python -m ...`. Use `.venv/bin/python` for GameTheory commands when it is available.

## Main runner

### run_all_figures.sh
Input: `--output-root DIR`, `--python BIN`, `--game-theory-only`, `--weight-sweep-run-config PATH`, `--skip-weight-sweep`.

Produces: one campaign folder under `script/simulations/output/figure_runs/<timestamp>/` with `figures/basic_simulation/`, `figures/game_theory/`, `runs/game_theory/`, `logs/`, `simulation_parameters.json`, and `relaunch_command.sh`.

## BasicSimulation

### activity_const_heatmaps.py
Input: `--weights W...`, `--number-of-points INT`, `--dt FLOAT`, `--output-path PATH`.

Produces: one combined heatmap figure for the always-active one-population model. Default output: `script/simulations/BasicSimulation/output/sight_weight_sunset_heatmaps.png`.

### activity_const_spread.py
Input: `--weights W...`, `--sunset-values T...`, `--number-of-points INT`, `--dt FLOAT`, `--max-workers INT`, `--output-path PATH`.

Produces: one spread-versus-weight figure across the selected day-night regimes. Default output: `script/simulations/BasicSimulation/output/sight_weight_sunset_spread.png`.

### spread_sleep_pattern_diffusion.py
Input: `--weights W...`, `--diffusion-values D...`, `--t-sunset FLOAT`, `--number-of-points INT`, `--dt FLOAT`, `--max-workers INT`, `--case-timeout-seconds FLOAT`, `--output-path PATH`.

Produces: one three-panel figure comparing the spread indicator across activity patterns and diffusion levels. Default output: `script/simulations/BasicSimulation/output/spread_sleep_pattern_diffusion.png`.

## GameTheory

### run_payoff_pipeline.sh
Input: `--output-dir DIR`, optional `--run-config PATH`, `--python BIN`, `--mean-x-axes LIST`, `--mean-show-variance BOOL`, `--replicator-time-span FLOAT`, `--replicator-time-steps INT`, `--replicator-plot-style line|stacked`, `--payoff-mode MODE`, plus the matrix-generation flags accepted by `payoff_matrix.py`.

Produces: one full payoff-analysis folder with the payoff matrices, heatmaps, `case_payoffs.csv`, `run_config.json`, `mean_analysis/`, `payoff_minmax_maxmin.json`, `payoff_nash_equilibrium.json`, `replicator_analysis/strategy_frequencies.png`, `logs/`, and `pipeline_command.sh`.

### payoff_matrix.py
Input: `--output-dir DIR`, optional `--run-config PATH`, optional `--strategy-codes C1,C2,...`, optional `--heatmap-prey CODE` with `--heatmap-predator CODE`, and the two-population model parameters `--t-sunset`, `--weights W1 W2`, `--sight-radius`, `--prey-sight-radius`, `--predator-sight-radius`, `--smell-radius`, `--prey-smell-radius`, `--predator-smell-radius`, `--number-of-points`, `--dt`, `--number-of-cycles`, `--observation-window`, `--payoff-mode`, `--prey-growth`, `--predator-decay`, `--predation-rate`, `--conversion-rate`, `--chi11`, `--chi12`, `--chi21`, `--chi22`, `--diffusion D1 D2`, `--initial-centers X1 X2`, `--initial-width`, `--max-workers`.

Produces: a saved payoff run. Always writes `case_payoffs.csv` and `run_config.json`. In `overlap` mode it writes `payoff_matrix.csv` and `payoff_matrix.png`. In `population-integral` and `net-growth` modes it writes `payoff_matrix_prey.csv`, `payoff_matrix_predator.csv`, `payoff_matrix_prey.png`, and `payoff_matrix_predator.png`. It also writes `population_heatmaps/` for the simulated activity pairs.

### payoff_mean_analysis.py
Input: `--x-axis w1|w2|cycle1|cycle2`, `--payoff-dir PATH`, `--output PATH`, optional `--show-variance BOOL`, optional `--payoff-player auto|legacy|prey|predator`.

Produces: one mean-payoff plot, optionally with a variance panel, from a single payoff folder or a directory containing multiple payoff runs.

### payoff_minmax_maxmin.py
Input: `--payoff-matrix PATH`, optional `--output PATH`.

Produces: `payoff_minmax_maxmin.json` next to the source matrix or in the requested output path.

### payoff_nash_equilibrium.py
Input: `--payoff-matrix PATH`, optional `--output PATH`.

Produces: `payoff_nash_equilibrium.json` containing the mixed Nash equilibria and expected payoffs.

### payoff_replicator_analysis.py
Input: `--payoff-matrix PATH`, `--time-span FLOAT`, `--time-steps INT`, `--output-dir DIR`, `--plot-style line|stacked`.

Produces: a replicator-dynamics figure set in the requested directory. The main artifact is `strategy_frequencies.png`.

### payoff_initial_condition_comparison.py
Input: optional `--run-config PATH`, `--conditions NAME...`, `--weight FLOAT`, `--t-sunset FLOAT`, `--number-of-points INT`, `--dt FLOAT`, `--number-of-cycles INT`, `--observation-window FLOAT`, `--diffusion D1 D2`, `--initial-width FLOAT`, `--profile-points INT`, `--cache-dir DIR`, `--output-path PATH`, `--max-workers INT`.

Produces: one comparison figure for the selected initial-condition families and a cache folder with one saved payoff run per condition.

### payoff_weight_nash_heatmap.py
Input: optional `--run-config PATH`, `--w1-values W...`, `--w2-values W...`, `--output-dir DIR`, optional `--strategy-codes C1,C2,...`, `--payoff-mode MODE`, the same two-population solver and reaction flags as `payoff_matrix.py`, `--max-workers INT`, and optional `--plot-only` to rebuild figures from an existing summary.

Produces: `nash_weight_summary.csv`, `nash_weight_details.json`, `nash_consensus_components.png`, `nash_equilibrium_diagnostics.png`, `run_config.json`, and one saved payoff folder per weight pair under `weight_runs/`.

### payoff_weight_nash_strategy_table.py
Input: one positional `source` pointing to a weight-sweep folder or directly to `nash_weight_details.json`, plus optional `--output-csv PATH` and `--output-markdown PATH`.

Produces: `nash_strategy_frequency_summary.csv` and `nash_strategy_frequency_summary.md`.