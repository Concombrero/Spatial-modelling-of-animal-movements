# Day-Night Modeling

This folder contains the 1D periodic day-night solver and the scripts used to
generate payoff matrices, spread indicators, and evolutionary-game figures.

The core class is `DayNightModel1D` in `solver.py`. Shared plotting and spread
helpers used by the visualization scripts live in `common_utils.py`.

## Setup

Run the commands below from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

All plotting scripts use a non-interactive Matplotlib backend and save figures
to `script/reproduce_day_night/output/` by default.

## Circadian regime codes

The payoff and evolutionary scripts use the following activity codes:

| Code | Label | Active intervals in phase coordinates |
| --- | --- | --- |
| `D` | Diurnal | `[0.0, 0.5]` |
| `N` | Nocturnal | `[0.5, 1.0]` |
| `P1` | Polyphasic 1 | `[0.0, 0.25] + [0.5, 0.75]` |
| `P2` | Polyphasic 2 | `[0.25, 0.5] + [0.75, 1.0]` |
| `M1` | Matutinal 1 | `[0.0, 0.25] + [0.75, 1.0]` |
| `M2` | Matutinal 2 | `[0.25, 0.75]` |

## Main workflows

### 1. Compute one payoff matrix

`payoff_matrix.py` computes the 6x6 prey-predator payoff data from the final
observation window.

```bash
python3 script/reproduce_day_night/payoff_matrix.py
```

Default outputs:

- `script/reproduce_day_night/output/Pay-off/payoff_matrix.csv`
- `script/reproduce_day_night/output/Pay-off/case_payoffs.csv`
- `script/reproduce_day_night/output/Pay-off/run_config.json`
- `script/reproduce_day_night/output/Pay-off/payoff_matrix.png`
- `script/reproduce_day_night/output/Pay-off/population_heatmaps/`

By default the script uses the legacy overlap payoff

$$
\int \sqrt{u_1 u_2}
$$

over the final observation window and writes the single predator payoff matrix
used by the zero-sum workflow.

You can switch to a population-specific payoff with:

```bash
python3 script/reproduce_day_night/payoff_matrix.py --payoff-mode population-integral
```

In that mode the script keeps the same observation window but writes two payoff
matrices instead:

- `script/reproduce_day_night/output/Pay-off/payoff_matrix_prey.csv`
- `script/reproduce_day_night/output/Pay-off/payoff_matrix_predator.csv`
- `script/reproduce_day_night/output/Pay-off/payoff_matrix_prey.png`
- `script/reproduce_day_night/output/Pay-off/payoff_matrix_predator.png`

where the prey payoff is based on

$$
\int u_1
$$

and the predator payoff is based on

$$
\int u_2.
$$

Useful variations:

```bash
python3 script/reproduce_day_night/payoff_matrix.py --weights 0.25 0.75
python3 script/reproduce_day_night/payoff_matrix.py --t-sunset 0.7
python3 script/reproduce_day_night/payoff_matrix.py \
  --prey-sight-radius 0.08 --predator-sight-radius 0.14 \
  --prey-smell-radius 0.18 --predator-smell-radius 0.25
python3 script/reproduce_day_night/payoff_matrix.py --heatmap-prey D --heatmap-predator N
```

### 2. Sweep the payoff matrix over sight weights

`payoff_matrix_weight_sweep.py` repeats the payoff workflow over a grid of
weight pairs.

```bash
python3 script/reproduce_day_night/payoff_matrix_weight_sweep.py --weights 0.25 0.5 0.75
```

Notes:

- `--weights` and `--weight-values` are equivalent.
- `--w1-values` and `--w2-values` can override each axis independently.
- `--sight-radius` and `--smell-radius` remain shared shorthands; use the
  prey/predator-specific radius flags to split the two populations.
- `--skip-existing` reuses a completed folder only when the saved run
  configuration matches the current parameters.

Example:

```bash
python3 script/reproduce_day_night/payoff_matrix_weight_sweep.py \
    --w1-values 0.7 \
    --weights 0.25 0.5 0.75 \
    --skip-existing
```

### 3. Analyze saved payoff outputs

`payoff_mean_analysis.py` reads either a single payoff run directory or a
parent directory containing many runs and plots the mean payoff against one
parameter.

```bash
python3 script/reproduce_day_night/payoff_mean_analysis.py \
    --x-axis w1 \
    --payoff-dir script/reproduce_day_night/output/Pay-off/weight_sweep \
    --output script/reproduce_day_night/output/Pay-off/mean_vs_w1.png \
    --show-variance true
```

For population-specific payoff runs, choose which matrix to average:

```bash
python3 script/reproduce_day_night/payoff_mean_analysis.py \
  --x-axis cycle1 \
  --payoff-dir script/reproduce_day_night/output/Pay-off \
  --output script/reproduce_day_night/output/Pay-off/mean_vs_cycle1_prey.png \
  --payoff-player prey
```

Supported x-axis values are `w1`, `w2`, `cycle1`, and `cycle2`.

### 4. Compute minmax, maxmin, and Nash equilibria from one payoff matrix

`payoff_minmax_maxmin.py` loads either a saved `payoff_matrix.csv` or a payoff
run directory. For the legacy overlap workflow it computes the prey minmax
value and the predator maxmin value. For the population-specific payoff mode it
computes each player's own maximin security value from its own matrix and saves
the result as JSON in the same folder.

```bash
python3 script/reproduce_day_night/payoff_minmax_maxmin.py \
  --payoff-matrix script/reproduce_day_night/output/Pay-off
```

Default output:

- `script/reproduce_day_night/output/Pay-off/payoff_minmax_maxmin.json`

`payoff_nash_equilibrium.py` loads the same source, builds the corresponding
zero-sum or two-matrix prey-predator game, computes Nash equilibria with
Nashpy, and saves the equilibria as JSON in the same folder.

```bash
python3 script/reproduce_day_night/payoff_nash_equilibrium.py \
  --payoff-matrix script/reproduce_day_night/output/Pay-off
```

Default output:

- `script/reproduce_day_night/output/Pay-off/payoff_nash_equilibrium.json`

`payoff_replicator_analysis.py` loads a saved `payoff_matrix.csv` or a payoff
run directory, prints Nash equilibria with Nashpy, and plots asymmetric
prey/predator replicator dynamics from the uniform initial condition.

```bash
python3 script/reproduce_day_night/payoff_replicator_analysis.py \
  --payoff-matrix script/reproduce_day_night/output/Pay-off
```

Default outputs:

- `script/reproduce_day_night/output/Pay-off/replicator_analysis/prey_strategy_frequencies.png`
- `script/reproduce_day_night/output/Pay-off/replicator_analysis/predator_strategy_frequencies.png`

### 5. Run the evolutionary game

`evolutionary_game.py` evolves shares of the six circadian regimes for prey and
predator populations while keeping ecological parameters fixed during each PDE
round.

```bash
python3 script/reproduce_day_night/evolutionary_game.py \
    --w1 0.5 \
    --w2 0.5 \
    --rounds 8 \
    --selection-events 2 \
    --selection-percentage 10
```

Default outputs:

- `script/reproduce_day_night/output/evolutionary_game/run_config.json`
- `script/reproduce_day_night/output/evolutionary_game/round_payoffs.csv`
- `script/reproduce_day_night/output/evolutionary_game/distribution_history.csv`
- `script/reproduce_day_night/output/evolutionary_game/selection_events.csv`
- `script/reproduce_day_night/output/evolutionary_game/strategy_shares.png`

The prey objective is to minimize overlap payoff. The predator objective is to
maximize it. Every regime is kept above a 1% share floor.

### 6. Sweep the evolutionary game over weights and radii

`evolutionary_game_parameter_sweep.py` runs the evolutionary game over the
Cartesian product of `w1`, `w2`, `R_smell_1`, `R_sight_1`, `R_smell_2`, and
`R_sight_2`.

Independent parameter cases can run in parallel with `--max-workers`.

```bash
python3 script/reproduce_day_night/evolutionary_game_parameter_sweep.py \
    --w1-values 0.25 0.5 \
    --w2-values 0.25 0.5 \
    --r-smell-1-values 0.2 \
    --r-sight-1-values 0.1 \
    --r-smell-2-values 0.2 \
  --r-sight-2-values 0.1 \
  --max-workers 4
```

Range syntax is also supported for every swept parameter:

```bash
python3 script/reproduce_day_night/evolutionary_game_parameter_sweep.py \
    --w1-range 0.2 0.8 0.1 \
    --w2-range 0.2 0.8 0.1 \
    --r-smell-1-range 0.15 0.25 0.05 \
    --r-sight-1-range 0.08 0.16 0.04 \
    --r-smell-2-range 0.15 0.25 0.05 \
    --r-sight-2-range 0.08 0.16 0.04
```

By default, this sweep only keeps one share plot per parameter set in
`script/reproduce_day_night/output/evolutionary_game_parameter_sweep/`.
When `--echo-round-progress` is combined with more than one worker, round logs
from different cases can interleave on stdout.

## Visualization scripts

These scripts focus on single-population spread or heatmap figures.

| Script | Purpose | Default output |
| --- | --- | --- |
| `activity_const_heatmaps.py` | Heatmaps for an always-active population under full day, half day, and full night | `output/sight_weight_sunset_heatmaps.png` |
| `activity_const_spread.py` | Normalized spread indicator $\Omega$ versus sight weight for the same lighting regimes | `output/sight_weight_sunset_spread.png` |
| `sleep_pattern_heatmaps.py` | Heatmaps for diurnal, nocturnal, polyphasic, and matutinal schedules | `output/sleep_pattern_heatmaps.png` |
| `sleep_pattern_spread.py` | Spread indicator $\Psi$ versus sight weight for several sleep schedules and several `t_sunset` values | `output/sleep_pattern_spread.png` |
| `spread_diurnal_vs_nocturnal.py` | Direct $\Psi$ comparison between diurnal and nocturnal schedules at fixed `t_sunset = 0.5` | `output/spread_diurnal_vs_nocturnal.png` |
| `spread_polyphasic_matutinal.py` | $\Psi$ comparison for polyphasic and matutinal schedules across several `t_sunset` values | `output/spread_polyphasic_matutinal.png` |

Example commands:

```bash
python3 script/reproduce_day_night/activity_const_heatmaps.py --weights 0 0.5 1
python3 script/reproduce_day_night/sleep_pattern_spread.py --weights 0 0.5 1 --sunset-values 0.25 0.5 0.75
python3 script/reproduce_day_night/spread_diurnal_vs_nocturnal.py --weights 0 0.5 1
```

## Minimal solver example

The example below assumes your working directory is
`script/reproduce_day_night/` or that this folder is on `PYTHONPATH`.

```python
import numpy as np

from solver import DayNightModel1D


def initial_condition(x):
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    centers = [0.25, 0.70]

    profiles = []
    for center in centers:
        wrapped = ((x - center + 0.5 * length) % length) - 0.5 * length
        profiles.append(np.exp(-0.5 * (wrapped / 0.08) ** 2))

    values = np.column_stack(profiles)
    masses = dx * np.sum(values, axis=0, keepdims=True)
    return values / masses


model = DayNightModel1D(
    a_border=0.0,
    b_border=1.0,
    number_of_points=128,
    total_time=2.0,
    dt=1.0e-3,
    initial_condition=initial_condition,
    number_of_population=2,
    coefficient_attraction=np.array([
        [0.12, -0.04],
        [0.06, 0.10],
    ]),
    coefficient_diffusion=np.array([0.05, 0.03]),
    cycle_period=1.0,
    time_input_mode="clock",
    day_start=8.0,
    day_end=22.0,
    activity_periods=[
        [(0.0, 8.0)],
        [(1.0, 6.0), (13.0, 22.0)],
    ],
    sight_weight=0.4,
    sight_radius=0.12,
    smell_radius=0.18,
)

time, solution = model.solve()
mass = model.get_mass()
overlap = model.get_overlap_energy(population_indices=(0, 1))
effective_parameters = model.get_effective_parameters(0.2)
```

## Key solver inputs

- `number_of_population`: number of interacting populations.
- `coefficient_attraction`: interaction matrix of shape `(m, m)`.
- `coefficient_diffusion`: diffusion vector of length `m`.
- `cycle_period`: duration of one full circadian cycle.
- `time_input_mode="phase"`: interpret day and activity intervals directly on
  the solver cycle.
- `time_input_mode="clock"`: convert clock-hour inputs into the repeated solver
  cycle.
- `activity_periods`: one list of active intervals per population. Multiple
  intervals are allowed.
- `initial_condition(x)`: for `m` populations, return an array of shape
  `(number_of_points, m)`.
- `sight_weight`, `sight_radius`, `smell_radius`: each can be a scalar or a
  per-population vector.
- `reaction_term(population, time, model)`: optional local source term with the
  same output shape as `population`.

## Useful solver methods

- `solve()`: run the simulation.
- `get_mass()`: return the mass of each population over time.
- `get_overlap_energy()`: compute the overlap functional over the final
  observation window.
- `get_effective_parameters(t)`: inspect the active diffusion, attraction, and
  activity mask at time `t`.
- `plot_solution_heatmaps(...)`: save one heatmap per population.
- `create_solution_gif(...)`: save an animation of the densities.

## Notes

- This folder is a collection of standalone scripts, not an installed Python
  package.
- For quick smoke tests, reduce `--number-of-points`, shorten
  `--number-of-cycles`, or increase `--dt`.
- The most useful entry points for new runs are usually `payoff_matrix.py` and
  `evolutionary_game.py`.