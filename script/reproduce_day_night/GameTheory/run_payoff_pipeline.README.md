# Payoff Pipeline Runner

`run_payoff_pipeline.sh` runs the full payoff workflow in one command inside a chosen output directory:

1. `payoff_matrix.py`
2. `payoff_mean_analysis.py`
3. `payoff_minmax_maxmin.py`
4. `payoff_nash_equilibrium.py`
5. `payoff_replicator_analysis.py`

The script also writes a `pipeline_command.sh` replay file and a `logs/` directory with one log per step.

Two payoff definitions are supported:

1. `overlap` for the historical payoff based on $\int \sqrt{u_1 u_2}$.
2. `population-integral` for a prey payoff based on $\int u_1$ and a predator payoff based on $\int u_2$ over the same final observation window.

## Location

- `script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh`

## Minimal usage

Run from the repository root:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/full_pipeline_run
```

Population-specific payoff version:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/full_pipeline_population_integral \
  --payoff-mode population-integral
```

## Examples

Shorter day length and weaker prey sight weight:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/short_day_run \
  --t-sunset 0.35 \
  --weights 0.2 0.7
```

Stronger predator visual advantage and longer replicator horizon:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/predator_day_advantage \
  --predator-sight-radius 0.18 \
  --prey-sight-radius 0.06 \
  --weights 0.3 0.9 \
  --replicator-time-span 120 \
  --replicator-time-steps 2400
```

Restrict mean-analysis plots to selected axes:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/cycle_focus \
  --mean-x-axes cycle1,cycle2 \
  --mean-show-variance true
```

Use a specific Python interpreter:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/venv_run \
  --python .venv/bin/python
```

## Output structure

The directory passed through `--output-dir` normally contains:

- `case_payoffs.csv`
- `run_config.json`
- `payoff_minmax_maxmin.json`
- `payoff_nash_equilibrium.json`
- `replicator_analysis/prey_strategy_frequencies.png`
- `replicator_analysis/predator_strategy_frequencies.png`
- `population_heatmaps/` when population heatmaps are requested or when a full payoff run is saved
- `logs/*.log`
- `pipeline_command.sh`

In `overlap` mode you also get:

- `payoff_matrix.csv`
- `payoff_matrix.png`
- `mean_analysis/mean_vs_*.png`

In `population-integral` mode you get instead:

- `payoff_matrix_prey.csv`
- `payoff_matrix_predator.csv`
- `payoff_matrix_prey.png`
- `payoff_matrix_predator.png`
- `mean_analysis/mean_vs_*_prey.png`
- `mean_analysis/mean_vs_*_predator.png`

## Main options

### General options

- `--output-dir DIR`: main output directory. Required.
- `--python BIN`: Python executable to use. Default: `python`.
- `--mean-x-axes LIST`: comma-separated mean-analysis axes. Useful values: `w1`, `w2`, `cycle1`, `cycle2`.
- `--mean-show-variance BOOL`: `true` or `false`.
- `--replicator-time-span FLOAT`: final time for the replicator dynamics.
- `--replicator-time-steps INT`: number of output samples for the replicator dynamics.
- `--replicator-plot-style STYLE`: `line` or `stacked`.
- `--payoff-mode MODE`: `overlap` or `population-integral`.

### Day-night and perception parameters

- `--t-sunset FLOAT`: daylight share in one cycle, in `[0, 1]`.
- `--weights W1 W2`: sight weights for prey and predator.
- `--sight-radius FLOAT`: shared sight-radius shorthand.
- `--prey-sight-radius FLOAT`: prey sight radius.
- `--predator-sight-radius FLOAT`: predator sight radius.
- `--smell-radius FLOAT`: shared smell-radius shorthand.
- `--prey-smell-radius FLOAT`: prey smell radius.
- `--predator-smell-radius FLOAT`: predator smell radius.

Practical interpretation:

- Lower `t-sunset` shortens the day and can weaken `D`.
- Lower `w1` or `prey-sight-radius` reduces the prey visual advantage.
- Higher `w2` or `predator-sight-radius` increases daytime predator pressure.
- Stronger smell contribution tends to smooth the day-night asymmetry.

### PDE discretization and horizon

- `--number-of-points INT`
- `--dt FLOAT`
- `--number-of-cycles INT`
- `--observation-window FLOAT`

### Lotka-Volterra reaction parameters

- `--prey-growth FLOAT`
- `--predator-decay FLOAT`
- `--predation-rate FLOAT`
- `--conversion-rate FLOAT`

### Attraction and diffusion parameters

- `--chi11 FLOAT`
- `--chi12 FLOAT`
- `--chi21 FLOAT`
- `--chi22 FLOAT`
- `--diffusion D1 D2`

### Initial condition parameters

- `--initial-centers X1 X2`
- `--initial-width FLOAT`

### Advanced outputs and parallelism

- `--strategy-codes LIST`: comma-separated subset of activity codes.
- `--heatmap-prey CODE`
- `--heatmap-predator CODE`
- `--max-workers INT`

## Notes

- The selected Python environment must contain at least `numpy`, `matplotlib`, `scipy`, and `nashpy`.
- The `nash` and `replicator` stages depend on `nashpy`.
- When the pipeline is run on a single payoff directory, `cycle1` and `cycle2` are often more informative than `w1` and `w2` for mean-analysis plots.