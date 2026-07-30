# Day-Night Modeling

This folder now separates the day-night codebase into three focused groups:

- `Solver/`: the `DayNightModel1D` solver and shared numerical utilities.
- `BasicSimulation/`: single-population heatmap and spread scripts.
- `GameTheory/`: payoff generation, payoff analysis, replicator dynamics, and the payoff pipeline runner.

Historical output artifacts are still kept under `script/reproduce_day_night/output/`. New default outputs now live inside the folder that owns each workflow.

## Setup

Run from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The game-theory workflows require `nashpy`. In this workspace the safest choice is usually:

```bash
.venv/bin/python
```

All plotting scripts use the non-interactive Matplotlib `Agg` backend.

## Folder layout

| Folder | Purpose | Default output root |
| --- | --- | --- |
| `Solver/` | PDE solver and shared helper functions | `script/reproduce_day_night/Solver/output/` |
| `BasicSimulation/` | heatmaps and spread indicators for one-population scenarios | `script/reproduce_day_night/BasicSimulation/output/` |
| `GameTheory/` | payoff matrices, minmax/maxmin, Nash, replicator, story search, pipeline | `script/reproduce_day_night/GameTheory/output/` |

## Circadian regime codes

| Code | Label | Active intervals in phase coordinates |
| --- | --- | --- |
| `D` | Diurnal | `[0.0, 0.5]` |
| `N` | Nocturnal | `[0.5, 1.0]` |
| `P1` | Polyphasic 1 | `[0.0, 0.25] + [0.5, 0.75]` |
| `P2` | Polyphasic 2 | `[0.25, 0.5] + [0.75, 1.0]` |
| `M` | Matutinal | `[0.0, 0.25] + [0.75, 1.0]` |
| `V` | Vespertine | `[0.25, 0.75]` |

## Common workflows

### Compute one payoff matrix

```bash
python3 script/reproduce_day_night/GameTheory/payoff_matrix.py
```

Default outputs:

- `script/reproduce_day_night/GameTheory/output/payoff/payoff_matrix.csv`
- `script/reproduce_day_night/GameTheory/output/payoff/case_payoffs.csv`
- `script/reproduce_day_night/GameTheory/output/payoff/run_config.json`
- `script/reproduce_day_night/GameTheory/output/payoff/payoff_matrix.png`
- `script/reproduce_day_night/GameTheory/output/payoff/population_heatmaps/`

Population-specific payoff mode:

```bash
python3 script/reproduce_day_night/GameTheory/payoff_matrix.py \
  --payoff-mode population-integral
```

### Run the full payoff pipeline

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/full_pipeline_run \
  --python .venv/bin/python
```

See `script/reproduce_day_night/GameTheory/run_payoff_pipeline.README.md` for the full pipeline reference.

### Analyze a saved payoff run

```bash
python3 script/reproduce_day_night/GameTheory/payoff_mean_analysis.py \
  --x-axis cycle1 \
  --payoff-dir script/reproduce_day_night/GameTheory/output/payoff \
  --output script/reproduce_day_night/GameTheory/output/payoff/mean_vs_cycle1.png \
  --show-variance true
```

```bash
python3 script/reproduce_day_night/GameTheory/payoff_minmax_maxmin.py \
  --payoff-matrix script/reproduce_day_night/GameTheory/output/payoff
```

```bash
python3 script/reproduce_day_night/GameTheory/payoff_nash_equilibrium.py \
  --payoff-matrix script/reproduce_day_night/GameTheory/output/payoff
```

```bash
python3 script/reproduce_day_night/GameTheory/payoff_replicator_analysis.py \
  --payoff-matrix script/reproduce_day_night/GameTheory/output/payoff
```

### Generate basic simulation figures

```bash
python3 script/reproduce_day_night/BasicSimulation/activity_const_heatmaps.py --weights 0 0.5 1
python3 script/reproduce_day_night/BasicSimulation/sleep_pattern_spread.py --weights 0 0.5 1 --sunset-values 0.25 0.5 0.75
python3 script/reproduce_day_night/BasicSimulation/spread_diurnal_vs_nocturnal.py --weights 0 0.5 1
```

Default outputs are written under:

- `script/reproduce_day_night/Solver/output/` for solver-generated figures and GIFs
- `script/reproduce_day_night/BasicSimulation/output/`
- `script/reproduce_day_night/GameTheory/output/` for game-theory workflows, including `story_search/`

## Minimal solver example

```python
import numpy as np

from script.reproduce_day_night.Solver import DayNightModel1D


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
- `time_input_mode="phase"`: interpret day and activity intervals directly on the solver cycle.
- `time_input_mode="clock"`: convert clock-hour inputs into the repeated solver cycle.
- `activity_periods`: one list of active intervals per population. Multiple intervals are allowed.
- `initial_condition(x)`: for `m` populations, return an array of shape `(number_of_points, m)`.
- `sight_weight`, `sight_radius`, `smell_radius`: each can be a scalar or a per-population vector.
- `reaction_term(population, time, model)`: optional local source term with the same output shape as `population`.

## Notes

- Scripts can be launched directly by path from the repository root.
- Default saved artifacts now stay inside the owning folder: `Solver/output/`, `BasicSimulation/output/`, and `GameTheory/output/`.
- The payoff pipeline uses module execution internally so the package layout stays consistent.
- For quick smoke tests, reduce `--number-of-points`, shorten `--number-of-cycles`, or increase `--dt`.