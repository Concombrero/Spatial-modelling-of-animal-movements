# Day-Night Solver

This folder contains a 1D multi-population day-night solver.

The solver class is `DayNightModel1D` in `solver.py`.

## Quick Use

Run the example script:

```bash
python3 script/reproduce_day_night/main.py
```

This creates outputs in `script/reproduce_day_night/output/`.

## Minimal Example

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
parameters = model.get_effective_parameters(0.2)
```

## Important Inputs

- `number_of_population`: number of populations.
- `coefficient_attraction`: interaction matrix of shape `(m, m)`.
- `coefficient_diffusion`: diffusion vector of length `m`.
- `cycle_period`: simulation duration of one full 24-hour cycle.
- `time_input_mode="clock"`: interpret `day_start`, `day_end`, and `activity_periods` as clock times in hours.
- `day_start`, `day_end`: daytime clock values. With `cycle_period=1.0`, `day_start=8`, `day_end=22` gives day on `[0, 14/24]`.
- `activity_periods`: one list of active clock intervals per population, for example `[[ (0, 8) ], [ (1, 6), (13, 22) ]]`.
- `initial_condition(x)`: should return one density per population.

For several populations, `initial_condition(x)` should return an array of shape `(number_of_points, number_of_population)`.

## Useful Methods

- `solve()`: runs the simulation.
- `get_mass()`: returns the mass of each population over time.
- `get_effective_parameters(t)`: returns the active diffusion, attraction, and per-population activity mask at time `t`.
- `plot_solution_heatmaps(...)`: plots one heatmap per population.
- `create_solution_gif(...)`: creates an animation of the densities.

## Notes

- In `time_input_mode="clock"`, one cycle still repeats forever, but the solver converts clock times into the matching interval inside each cycle.
- During inactive periods, the diffusion of an inactive population is zero and that population's attraction row is set to zero.
- The solver supports several cycles by setting `total_time` larger than `cycle_period`.