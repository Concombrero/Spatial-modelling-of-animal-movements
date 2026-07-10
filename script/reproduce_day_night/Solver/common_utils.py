import numpy as np


DEFAULT_INITIAL_CENTER = 0.5
DEFAULT_INITIAL_WIDTH = 0.08
DEFAULT_SIGHT_RADIUS = 0.1
DEFAULT_SMELL_RADIUS = 0.2
SPREAD_NORMALIZATION_FACTOR = 12.0


def gaussian_initial_condition(
    x,
    center=DEFAULT_INITIAL_CENTER,
    width=DEFAULT_INITIAL_WIDTH,
):
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
    values = np.exp(-0.5 * (wrapped_distance / width) ** 2)
    return values / (dx * np.sum(values))


def build_periodic_squared_distance_matrix(x, length):
    wrapped_distances = (
        (x[np.newaxis, :] - x[:, np.newaxis] + 0.5 * length) % length
    ) - 0.5 * length
    return wrapped_distances**2


def integrate_trapezoid(values, time_grid):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=time_grid)
    return np.trapz(values, x=time_grid)


def compute_spread_indicator(model, observation_window, population_index=0):
    if not 0 <= population_index < model.number_of_population:
        raise IndexError("population_index is out of bounds.")

    window_start = max(model.time[-1] - observation_window, model.time[0])
    window_mask = model.time >= (window_start - 1.0e-12)
    window_time = model.time[window_mask]
    if window_time.size == 0:
        raise ValueError("The observation window does not overlap the solver time grid.")

    window_density = model.U[window_mask, :, population_index]
    masses = model.dx * np.sum(window_density, axis=1, keepdims=True)
    if np.any(masses <= 0.0):
        raise ValueError(
            "Cannot normalize the density because the observation window contains non-positive mass."
        )

    normalised_density = window_density / masses
    squared_distance_matrix = build_periodic_squared_distance_matrix(
        model.x,
        model.length,
    )
    centred_second_moments = model.dx * normalised_density @ squared_distance_matrix.T
    minimum_second_moments = np.min(centred_second_moments, axis=1)

    spread_indicator = (
        SPREAD_NORMALIZATION_FACTOR / (model.length**2)
    ) * integrate_trapezoid(
        minimum_second_moments,
        window_time,
    )
    return float(np.clip(spread_indicator, 0.0, 1.0))