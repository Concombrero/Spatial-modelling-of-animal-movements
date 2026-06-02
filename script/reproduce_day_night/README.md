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
    day_start=0.0,
    day_end=0.5,
    activity_start=0.05,
    activity_end=0.35,
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
- `cycle_period`: length of one day-night cycle.
- `day_start`, `day_end`: define the daytime interval inside one cycle.
- `activity_start`, `activity_end`: define when populations are active inside one cycle.
- `initial_condition(x)`: should return one density per population.

For several populations, `initial_condition(x)` should return an array of shape `(number_of_points, number_of_population)`.

## Useful Methods

- `solve()`: runs the simulation.
- `get_mass()`: returns the mass of each population over time.
- `get_effective_parameters(t)`: returns the active diffusion and attraction at time `t`.
- `plot_solution_heatmaps(...)`: plots one heatmap per population.
- `create_solution_gif(...)`: creates an animation of the densities.

## Notes

- During inactive periods, diffusion and attraction are both zero.
- The solver supports several cycles by setting `total_time` larger than `cycle_period`.