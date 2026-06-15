# Day-Night Solver

This folder contains a 1D multi-population day-night solver.

The solver class is `DayNightModel1D` in `solver.py`.

## Quick Use

Run the example script:

```bash
python3 script/reproduce_day_night/main.py
```

This creates outputs in `script/reproduce_day_night/output/`.

To compute the 6x6 prey-predator payoff matrix defined in the paper at
`t_sunset = 0.5` with default `w_1 = w_2 = 0.5`, run:

```bash
python3 script/reproduce_day_night/payoff_matrix.py
```

To choose a different sight weight for each population, pass two values:

```bash
python3 script/reproduce_day_night/payoff_matrix.py --weights 0.25 0.75
```

The payoff script defaults to a coarser grid (`64` points, `dt = 0.1`) so the
full 6x6 matrix remains practical to compute. Use `--number-of-points` and
`--dt` to refine the simulation if needed.

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
- `sight_weight`: either one scalar shared by every population or one value per population.
- `reaction_term`: optional callable `reaction_term(population, time, model)` returning one source term per population.

For several populations, `initial_condition(x)` should return an array of shape `(number_of_points, number_of_population)`.

When `reaction_term` is provided, the callback receives:

- `population`: the current state with shape `(number_of_points, number_of_population)`.
- `time`: the current simulation time.
- `model`: the current `DayNightModel1D` instance, which gives access to helpers such as `is_active(...)` and `get_activity_mask(...)`.

The callback must return an array with the same shape as `population`.

## Reaction-Term Example

The solver can now include local reaction terms in addition to diffusion and nonlocal advection. For the modified Lotka-Volterra system from the paper,

```python
import numpy as np

from solver import DayNightModel1D


def modified_lotka_volterra(population, time, model):
    u1 = population[:, 0]
    u2 = population[:, 1]

    active_1 = float(model.is_active(time, population_index=0))
    active_2 = float(model.is_active(time, population_index=1))

    r1 = 0.6
    r2 = 0.4
    a = 0.8
    b = 0.5

    return np.column_stack(
        (
            active_1 * r1 * u1 - active_2 * a * u1 * u2,
            -r2 * u2 + active_2 * b * u1 * u2,
        )
    )


model = DayNightModel1D(
    a_border=0.0,
    b_border=1.0,
    number_of_points=128,
    total_time=2.0,
    dt=1.0e-3,
    number_of_population=2,
    coefficient_attraction=np.array([
        [0.12, -0.04],
        [0.06, 0.10],
    ]),
    coefficient_diffusion=np.array([0.05, 0.03]),
    cycle_period=1.0,
    activity_periods=[
        [(0.0, 0.5)],
        [(0.5, 1.0)],
    ],
    sight_weight=[0.4, 0.7],
    sight_radius=0.12,
    smell_radius=0.18,
    reaction_term=modified_lotka_volterra,
)

time, solution = model.solve()
```

This implements

```text
f_1(u_1, u_2, t) = 1_{T_active,1}(t) r_1 u_1 - 1_{T_active,2}(t) a u_1 u_2
f_2(u_1, u_2, t) = -r_2 u_2 + 1_{T_active,2}(t) b u_1 u_2
```

through the solver's built-in activity schedule. The reaction callback is evaluated at every RK4 stage, so it should be deterministic and free of side effects.

## Useful Methods

- `solve()`: runs the simulation.
- `get_mass()`: returns the mass of each population over time.
- `get_overlap_energy()`: computes $\int_T^{T+\tau}\int_\Omega \sqrt{u_i(x,t)u_j(x,t)}\,dx\,dt$ over the final observation window, with `tau=cycle_period` by default.
- `get_effective_parameters(t)`: returns the active diffusion, attraction, and per-population activity mask at time `t`.
- `plot_solution_heatmaps(...)`: plots one heatmap per population.
- `create_solution_gif(...)`: creates an animation of the densities.

## Notes

- In `time_input_mode="clock"`, one cycle still repeats forever, but the solver converts clock times into the matching interval inside each cycle.
- During inactive periods, the diffusion of an inactive population is zero and that population's attraction row is set to zero.
- When `reaction_term` is set, the total mass is allowed to grow or decay. The solver only clips small negative values and rescales back to the RK4 candidate mass for that step.
- The solver supports several cycles by setting `total_time` larger than `cycle_period`.