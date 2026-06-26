import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class DayNightModel1D:
    """One-dimensional spectral solver for a multi-population day-night model."""

    def __init__(
        self,
        a_border,
        b_border,
        number_of_points,
        total_time,
        dt,
        initial_condition=None,
        *,
        coefficient_attraction,
        coefficient_diffusion,
        cycle_period,
        number_of_population=1,
        day_start=0.0,
        day_end=None,
        time_input_mode="phase",
        clock_hours_per_cycle=24.0,
        activity_mode="always",
        activity_start=None,
        activity_end=None,
        activity_periods=None,
        sight_weight=0.5,
        sight_radius=0.05,
        smell_radius=0.15,
        reaction_term=None,
    ):
        self.a_border = float(a_border)
        self.b_border = float(b_border)
        self.number_of_points = int(number_of_points)
        self.total_time = float(total_time)
        self.dt = float(dt)
        self.initial_condition = initial_condition
        self.number_of_population = int(number_of_population)
        self.coefficient_attraction = self._coerce_attraction_matrix(
            coefficient_attraction
        )
        self.coefficient_diffusion = self._coerce_diffusion_vector(
            coefficient_diffusion
        )
        self.cycle_period = float(cycle_period)
        self.time_input_mode = str(time_input_mode)
        self.clock_hours_per_cycle = float(clock_hours_per_cycle)
        self._validate_time_scale_inputs()
        self.day_start, self.day_end = self._resolve_day_interval(day_start, day_end)
        self.day_duration = self._compute_interval_duration(
            self.day_start,
            self.day_end,
        )
        self.activity_mode = str(activity_mode)
        self.activity_intervals = self._resolve_activity_intervals(
            activity_periods,
            activity_start,
            activity_end,
        )
        self.sight_weight = self._coerce_population_parameter(
            sight_weight,
            argument_name="sight_weight",
        )
        self.sight_radius = self._coerce_population_parameter(
            sight_radius,
            argument_name="sight_radius",
        )
        self.smell_radius = self._coerce_population_parameter(
            smell_radius,
            argument_name="smell_radius",
        )
        self.reaction_term = reaction_term

        self.length = self._compute_domain_length()
        self.number_of_steps = self._compute_number_of_steps()

        self._validate_inputs()

        self.dx = self._compute_dx()
        self.x = self._build_space_grid()
        self.time = self._build_time_grid()
        self.time_steps = self.number_of_steps
        self.wavenumbers = self._build_wavenumbers()
        self.rk4_substeps = self._compute_rk4_substeps()
        self.internal_dt = self.dt / self.rk4_substeps

        self.kernel_grid = self._build_kernel_grid()
        self.smell_kernel_values = self._generate_smell_kernel_values()
        self.sight_kernel_values = self._generate_sight_kernel_values()
        self.smell_kernel_fourier = self._fft_vector(self.smell_kernel_values)
        self.sight_kernel_fourier = self._fft_vector(self.sight_kernel_values)

        self.U = self._initialise_solution_storage()
        self.U[0, :, :] = self._evaluate_initial_condition()

        self.U_fourier = np.zeros_like(self.U, dtype=complex)
        self.U_fourier[0, :, :] = self._fft_matrix(self.U[0, :, :])
        self._solution_computed = False

    def _coerce_attraction_matrix(self, values):
        array = np.asarray(values, dtype=float)
        if self.number_of_population == 1 and array.shape in ((), (1,)):
            return array.reshape(1, 1)
        return array

    def _coerce_diffusion_vector(self, values):
        array = np.asarray(values, dtype=float)
        if self.number_of_population == 1 and array.shape == ():
            return array.reshape(1)
        return array

    def _coerce_population_parameter(self, values, *, argument_name):
        array = np.asarray(values, dtype=float)
        if array.shape == ():
            return np.full(self.number_of_population, float(array), dtype=float)

        flattened = np.ravel(array)
        if flattened.shape == (self.number_of_population,):
            return flattened.astype(float, copy=False)

        raise ValueError(
            f"{argument_name} must be a scalar or have length number_of_population."
        )

    def _compute_domain_length(self):
        return self.b_border - self.a_border

    def _compute_number_of_steps(self):
        return int(round(self.total_time / self.dt))

    def _compute_wrapped_duration(self, raw_duration, period, *, allow_full_cycle=False):
        wrapped_duration = np.mod(float(raw_duration), period)

        if allow_full_cycle and np.isclose(wrapped_duration, 0.0):
            cycle_count = float(raw_duration) / period
            if not np.isclose(cycle_count, 0.0) and np.isclose(
                cycle_count,
                round(cycle_count),
            ):
                return period

        return wrapped_duration

    def _validate_time_scale_inputs(self):
        if self.time_input_mode not in {"phase", "clock"}:
            raise ValueError("time_input_mode must be 'phase' or 'clock'.")

        if (
            not np.isfinite(self.clock_hours_per_cycle)
            or self.clock_hours_per_cycle <= 0.0
        ):
            raise ValueError("clock_hours_per_cycle must be positive.")

    def _resolve_day_interval(self, day_start, day_end):
        day_start = float(day_start)
        if day_end is None:
            default_duration = (
                0.5 * self.clock_hours_per_cycle
                if self.time_input_mode == "clock"
                else 0.5 * self.cycle_period
            )
            day_end = day_start + default_duration
        else:
            day_end = float(day_end)

        if self.time_input_mode == "phase":
            self._clock_reference_time = 0.0
            return day_start, day_end

        self._clock_reference_time = day_start
        day_duration = self._compute_clock_interval_duration(day_start, day_end)
        return 0.0, day_duration

    def _compute_clock_interval_duration(self, start, end, allow_full_cycle=False):
        wrapped_duration = self._compute_wrapped_duration(
            float(end) - float(start),
            self.clock_hours_per_cycle,
            allow_full_cycle=allow_full_cycle,
        )
        return self.cycle_period * wrapped_duration / self.clock_hours_per_cycle

    def _convert_clock_time_to_cycle_phase(self, value):
        return (
            self.cycle_period
            * np.mod(
                float(value) - self._clock_reference_time,
                self.clock_hours_per_cycle,
            )
            / self.clock_hours_per_cycle
        )

    def _resolve_interval(self, start, end, *, allow_full_cycle=False):
        if self.time_input_mode == "clock":
            return (
                self._convert_clock_time_to_cycle_phase(start),
                self._compute_clock_interval_duration(
                    start,
                    end,
                    allow_full_cycle=allow_full_cycle,
                ),
            )

        return (
            float(start),
            self._compute_interval_duration(
                start,
                end,
                allow_full_cycle=allow_full_cycle,
            ),
        )

    def _build_uniform_activity_intervals(self, interval):
        return [[interval] for _ in range(self.number_of_population)]

    def _is_interval_like(self, values):
        try:
            start, end = values
        except (TypeError, ValueError):
            return False

        return np.isscalar(start) and np.isscalar(end)

    def _coerce_population_activity_periods(self, activity_periods):
        population_periods = list(activity_periods)
        if self.number_of_population == 1:
            if not population_periods:
                population_periods = [[]]
            elif all(self._is_interval_like(interval) for interval in population_periods):
                population_periods = [population_periods]

        if len(population_periods) != self.number_of_population:
            raise ValueError(
                "activity_periods must provide one list of intervals per population."
            )

        schedules = []
        for population_index, intervals in enumerate(population_periods):
            try:
                interval_list = list(intervals)
            except TypeError as exc:
                raise ValueError(
                    "Each entry in activity_periods must be a list of (start, end) pairs."
                ) from exc

            population_schedule = []
            for interval in interval_list:
                if not self._is_interval_like(interval):
                    raise ValueError(
                        f"activity_periods[{population_index}] entries must be (start, end) pairs."
                    )

                start, end = interval
                interval_start, interval_duration = self._resolve_interval(
                    start,
                    end,
                    allow_full_cycle=True,
                )
                if np.isclose(interval_duration, 0.0):
                    raise ValueError(
                        "activity_periods must not contain zero-duration intervals."
                    )

                population_schedule.append((interval_start, interval_duration))

            schedules.append(population_schedule)

        return schedules

    def _resolve_activity_intervals(
        self,
        activity_periods,
        activity_start,
        activity_end,
    ):
        if activity_periods is not None:
            if activity_start is not None or activity_end is not None:
                raise ValueError(
                    "Provide activity_periods or activity_start/activity_end, not both."
                )
            return self._coerce_population_activity_periods(activity_periods)

        if (activity_start is None) != (activity_end is None):
            raise ValueError(
                "Provide both activity_start and activity_end, or neither."
            )

        if activity_start is not None:
            interval = self._resolve_interval(
                activity_start,
                activity_end,
                allow_full_cycle=True,
            )
            return self._build_uniform_activity_intervals(interval)

        if self.activity_mode == "diurnal":
            return self._build_uniform_activity_intervals(
                (self.day_start, self.day_duration)
            )

        if self.activity_mode == "nocturnal":
            return self._build_uniform_activity_intervals(
                (self.day_end, self.cycle_period - self.day_duration)
            )

        return self._build_uniform_activity_intervals((self.day_start, self.cycle_period))

    def _compute_interval_duration(self, start, end, allow_full_cycle=False):
        if not np.isfinite(self.cycle_period) or self.cycle_period <= 0.0:
            return np.nan

        return self._compute_wrapped_duration(
            float(end) - float(start),
            self.cycle_period,
            allow_full_cycle=allow_full_cycle,
        )

    def _validate_day_interval(self):
        if not np.isfinite(self.day_start) or not np.isfinite(self.day_end):
            raise ValueError("day_start and day_end must be finite.")

        if np.isclose(self.day_duration, 0.0):
            raise ValueError(
                "day_start and day_end must define non-zero day and night intervals."
            )

    def _validate_activity_intervals(self):
        for population_index, intervals in enumerate(self.activity_intervals):
            for interval_start, interval_duration in intervals:
                if not np.isfinite(interval_start) or not np.isfinite(interval_duration):
                    raise ValueError(
                        f"activity interval {population_index + 1} must be finite."
                    )

                if interval_duration < 0.0:
                    raise ValueError(
                        f"activity interval {population_index + 1} must be non-negative."
                    )

                if np.isclose(interval_duration, 0.0):
                    raise ValueError(
                        f"activity interval {population_index + 1} must be non-zero."
                    )

                if (
                    interval_duration > self.cycle_period
                    and not np.isclose(interval_duration, self.cycle_period)
                ):
                    raise ValueError(
                        f"activity interval {population_index + 1} exceeds one cycle."
                    )

    def _validate_inputs(self):
        if self.length <= 0.0:
            raise ValueError("b_border must be larger than a_border.")

        if self.number_of_population < 1:
            raise ValueError("number_of_population must be at least 1.")

        if self.number_of_points < 2:
            raise ValueError("number_of_points must be at least 2.")

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self.total_time < 0.0:
            raise ValueError("total_time must be non-negative.")

        if not np.isclose(self.number_of_steps * self.dt, self.total_time):
            raise ValueError("total_time must be an integer multiple of dt.")

        if self.coefficient_attraction.shape != (
            self.number_of_population,
            self.number_of_population,
        ):
            raise ValueError(
                "coefficient_attraction must have shape "
                "(number_of_population, number_of_population)."
            )

        if not np.all(np.isfinite(self.coefficient_attraction)):
            raise ValueError("coefficient_attraction must be finite.")

        if self.coefficient_diffusion.shape != (self.number_of_population,):
            raise ValueError(
                "coefficient_diffusion must have length number_of_population."
            )

        if not np.all(np.isfinite(self.coefficient_diffusion)):
            raise ValueError("coefficient_diffusion must be finite.")

        if np.any(self.coefficient_diffusion < 0.0):
            raise ValueError("coefficient_diffusion must be non-negative.")

        if not np.isfinite(self.cycle_period) or self.cycle_period <= 0.0:
            raise ValueError("cycle_period must be positive.")

        self._validate_day_interval()

        if self.activity_mode not in {"always", "diurnal", "nocturnal"}:
            raise ValueError(
                "activity_mode must be 'always', 'diurnal', or 'nocturnal'."
            )

        self._validate_activity_intervals()

        if not np.all(np.isfinite(self.sight_weight)):
            raise ValueError("sight_weight must be finite.")

        if np.any((self.sight_weight < 0.0) | (self.sight_weight > 1.0)):
            raise ValueError(
                "Each sight_weight must lie in the interval [0, 1]."
            )

        if not np.all(np.isfinite(self.sight_radius)):
            raise ValueError("sight_radius must be finite.")

        if np.any(self.sight_radius <= 0.0):
            raise ValueError("Each sight_radius must be positive.")

        if not np.all(np.isfinite(self.smell_radius)):
            raise ValueError("smell_radius must be finite.")

        if np.any(self.smell_radius <= 0.0):
            raise ValueError("Each smell_radius must be positive.")

        if self.reaction_term is not None and not callable(self.reaction_term):
            raise ValueError("reaction_term must be callable or None.")

    def _compute_dx(self):
        return self.length / self.number_of_points

    def _build_space_grid(self):
        return np.linspace(
            self.a_border,
            self.b_border,
            self.number_of_points,
            endpoint=False,
        )

    def _build_time_grid(self):
        return np.arange(self.number_of_steps + 1, dtype=float) * self.dt

    def _build_wavenumbers(self):
        return 2.0 * np.pi * np.fft.fftfreq(self.number_of_points, d=self.dx)

    def _compute_rk4_substeps(self):
        max_diffusion = float(np.max(self.coefficient_diffusion))
        max_wavenumber_squared = float(np.max(self.wavenumbers**2))
        stability_scale = self.dt * max_diffusion * max_wavenumber_squared
        stable_limit = 2.5

        if stability_scale <= stable_limit:
            return 1

        return int(np.ceil(stability_scale / stable_limit))

    def _coerce_population_state(self, population, *, argument_name="population"):
        values = np.asarray(population, dtype=float)

        if self.number_of_population == 1:
            if values.shape == (self.number_of_points,):
                return values[:, np.newaxis]

            if values.shape == (1, self.number_of_points):
                return values.T

        if values.shape == (self.number_of_population, self.number_of_points):
            return values.T

        if values.shape == (self.number_of_points, self.number_of_population):
            return values

        raise ValueError(
            f"{argument_name} must have shape "
            "(number_of_points, number_of_population) or "
            "(number_of_population, number_of_points)."
        )

    def _compute_advection_dt_limit(self, population, time):
        if np.allclose(self._effective_attraction(time), 0.0):
            return np.inf

        velocity = self._compute_advective_velocity(population, time)
        max_velocity = float(np.max(np.abs(velocity)))
        if max_velocity <= 0.0 or not np.isfinite(max_velocity):
            return np.inf

        cfl_number = 0.5
        return cfl_number * self.dx / max_velocity

    def _compute_compression_dt_limit(self, population, time):
        if np.allclose(self._effective_attraction(time), 0.0):
            return np.inf

        compression_rate = np.zeros_like(population)
        for population_index in range(self.number_of_population):
            smoothed_curvature = self._compute_smoothed_density_second_derivative(
                population,
                time,
                population_index=population_index,
            )
            compression_rate[:, population_index] = self._apply_interaction_matrix(
                smoothed_curvature,
                time,
                population_index=population_index,
            )

        max_compression_rate = float(np.max(np.abs(compression_rate)))
        if max_compression_rate <= 0.0 or not np.isfinite(max_compression_rate):
            return np.inf

        safety_factor = 0.25
        return safety_factor / max_compression_rate

    def _compute_positivity_dt_limit(self, population, time):
        rhs = self._compute_rhs(time, population)
        density_floor = np.maximum(
            1.0e-12,
            1.0e-5 * np.max(population, axis=0, keepdims=True),
        )
        unstable_mask = (rhs < 0.0) & (population > density_floor)
        if not np.any(unstable_mask):
            return np.inf

        depletion_times = population[unstable_mask] / (-rhs[unstable_mask])
        min_depletion_time = float(np.min(depletion_times))
        if min_depletion_time <= 0.0 or not np.isfinite(min_depletion_time):
            return np.inf

        safety_factor = 0.5
        return safety_factor * min_depletion_time

    def _compute_internal_step_size(self, population, time):
        diffusion_limited_dt = self.internal_dt
        advection_limited_dt = self._compute_advection_dt_limit(population, time)
        compression_limited_dt = self._compute_compression_dt_limit(
            population,
            time,
        )
        positivity_limited_dt = self._compute_positivity_dt_limit(
            population,
            time,
        )
        dt_step = min(
            diffusion_limited_dt,
            advection_limited_dt,
            compression_limited_dt,
            positivity_limited_dt,
        )

        if not np.isfinite(dt_step) or dt_step <= 0.0:
            return diffusion_limited_dt

        return dt_step

    def _initialise_solution_storage(self):
        return np.zeros(
            (self.number_of_steps + 1, self.number_of_points, self.number_of_population),
            dtype=float,
        )

    def _evaluate_initial_condition(self):
        if self.initial_condition is None:
            return self._build_default_initial_condition()

        values = self._coerce_population_state(
            self.initial_condition(self.x),
            argument_name="initial_condition(x)",
        )
        return self._normalise_population_profiles(values)

    def _build_default_initial_condition(self):
        if self.number_of_population == 1:
            centers = [self.a_border + 0.35 * self.length]
            gaussian_width = max(self.length / 12.0, self.dx)
        else:
            centers = self._build_default_population_centers()
            gaussian_width = self._compute_default_gaussian_width()

        profiles = [
            self._build_periodic_gaussian(center, gaussian_width)
            for center in centers
        ]
        values = np.column_stack(profiles)
        return self._normalise_population_profiles(values)

    def _build_default_population_centers(self):
        spacing = self.length / self.number_of_population
        return self.a_border + (0.5 + np.arange(self.number_of_population)) * spacing

    def _compute_default_gaussian_width(self):
        spacing = self.length / self.number_of_population
        return max(spacing / 6.0, self.dx)

    def _build_periodic_gaussian(self, center, width):
        wrapped_distance = (
            (self.x - center + 0.5 * self.length) % self.length
        ) - 0.5 * self.length
        return np.exp(-0.5 * (wrapped_distance / width) ** 2)

    def _normalise_population_profiles(self, values):
        values = np.asarray(values, dtype=float)
        masses = self.dx * np.sum(values, axis=0, keepdims=True)
        if np.any(masses <= 0.0):
            raise ValueError("Each population must have positive mass.")
        return values / masses

    def _build_kernel_grid(self):
        offsets = np.arange(self.number_of_points, dtype=float) * self.dx
        return np.where(offsets < 0.5 * self.length, offsets, offsets - self.length)

    def _smell_kernel(self, values, radius):
        values = np.asarray(values, dtype=float)
        radius = float(radius)
        denominator = radius * np.sqrt(2.0 * np.pi)
        return np.exp(-0.5 * (values / radius) ** 2) / denominator

    def _sight_kernel(self, values, radius):
        values = np.asarray(values, dtype=float)
        radius = float(radius)
        return 0.5 * np.exp(-np.abs(values) / radius) / radius

    def _normalise_kernel_values(self, kernel_values):
        normalisation = self.dx * np.sum(kernel_values)
        if normalisation <= 0.0:
            raise ValueError("Kernel must have positive mass.")
        return kernel_values / normalisation

    def _generate_population_kernel_values(self, radii, kernel_builder):
        kernel_columns = [
            self._normalise_kernel_values(
                kernel_builder(self.kernel_grid, float(radii[population_index]))
            )
            for population_index in range(self.number_of_population)
        ]
        if self.number_of_population == 1:
            return kernel_columns[0]
        return np.column_stack(kernel_columns)

    def _generate_smell_kernel_values(self):
        return self._generate_population_kernel_values(
            self.smell_radius,
            self._smell_kernel,
        )

    def _generate_sight_kernel_values(self):
        return self._generate_population_kernel_values(
            self.sight_radius,
            self._sight_kernel,
        )

    def _fft_vector(self, values):
        return np.fft.fft(values, axis=0)

    def _ifft_vector(self, values):
        return np.fft.ifft(values, axis=0).real

    def _fft_matrix(self, values):
        return np.fft.fft(values, axis=0)

    def _ifft_matrix(self, values):
        return np.fft.ifft(values, axis=0).real

    def _integrate_trapezoid(self, values, time_grid):
        if hasattr(np, "trapezoid"):
            return np.trapezoid(values, x=time_grid)
        return np.trapz(values, x=time_grid)

    def _first_derivative_multiplier(self):
        return 1j * self.wavenumbers

    def _second_derivative_multiplier(self):
        return -(self.wavenumbers**2)

    def _apply_first_derivative_fourier(self, values_hat):
        return self._first_derivative_multiplier()[:, np.newaxis] * values_hat

    def _apply_second_derivative_fourier(self, values_hat):
        return self._second_derivative_multiplier()[:, np.newaxis] * values_hat

    def _phase_in_cycle(self, time, reference_time=0.0):
        return np.mod(time - reference_time, self.cycle_period)

    def _is_within_interval(self, time, start, duration):
        if duration >= self.cycle_period:
            return True
        return bool(self._phase_in_cycle(time, start) < duration)

    def is_daytime(self, time):
        return self._is_within_interval(time, self.day_start, self.day_duration)

    def _is_active_in_intervals(self, time, intervals):
        return any(
            self._is_within_interval(time, interval_start, interval_duration)
            for interval_start, interval_duration in intervals
        )

    def get_activity_mask(self, time):
        return np.array(
            [
                self._is_active_in_intervals(time, intervals)
                for intervals in self.activity_intervals
            ],
            dtype=bool,
        )

    def is_active(self, time, population_index=None):
        activity_mask = self.get_activity_mask(time)
        if population_index is None:
            if self.number_of_population == 1:
                return bool(activity_mask[0])
            return bool(np.any(activity_mask))

        if not 0 <= population_index < self.number_of_population:
            raise IndexError("population_index is out of bounds.")

        return bool(activity_mask[population_index])

    def _activity_factor(self, time):
        return self.get_activity_mask(time).astype(float)

    def _effective_diffusion(self, time):
        return self.coefficient_diffusion * self._activity_factor(time)

    def _effective_attraction(self, time):
        return self.coefficient_attraction * self._activity_factor(time)[:, np.newaxis]

    def _sight_factor(self, time):
        return 1.0 if self.is_daytime(time) else 0.0

    def _blend_kernel_parts(
        self,
        smell_values,
        sight_values,
        time,
        population_index=None,
    ):
        sight_factor = self._sight_factor(time)

        if population_index is not None:
            if np.ndim(smell_values) == 2:
                smell_values = smell_values[:, population_index]
            if np.ndim(sight_values) == 2:
                sight_values = sight_values[:, population_index]
            sight_weight = float(self.sight_weight[population_index])
            smell_part = (1.0 - sight_weight) * smell_values
            sight_part = sight_weight * sight_factor * sight_values
            return smell_part, sight_part

        if self.number_of_population == 1:
            if np.ndim(smell_values) == 2:
                smell_values = smell_values[:, 0]
            if np.ndim(sight_values) == 2:
                sight_values = sight_values[:, 0]
            sight_weight = float(self.sight_weight[0])
            smell_part = (1.0 - sight_weight) * smell_values
            sight_part = sight_weight * sight_factor * sight_values
            return smell_part, sight_part

        if np.ndim(smell_values) == 1:
            smell_values = smell_values[:, np.newaxis]
        if np.ndim(sight_values) == 1:
            sight_values = sight_values[:, np.newaxis]
        weight_vector = self.sight_weight[np.newaxis, :]
        smell_part = smell_values * (1.0 - weight_vector)
        sight_part = sight_values * (sight_factor * weight_vector)
        return smell_part, sight_part

    def _combined_kernel_values(self, time, population_index=None):
        smell_part, sight_part = self._blend_kernel_parts(
            self.smell_kernel_values,
            self.sight_kernel_values,
            time,
            population_index=population_index,
        )
        return smell_part + sight_part

    def _combined_kernel_fourier(self, time, population_index=None):
        smell_part, sight_part = self._blend_kernel_parts(
            self.smell_kernel_fourier,
            self.sight_kernel_fourier,
            time,
            population_index=population_index,
        )
        return smell_part + sight_part

    def _convolve_in_fourier_space(self, values_hat, time, population_index=None):
        kernel_hat = self._combined_kernel_fourier(
            time,
            population_index=population_index,
        )
        if np.ndim(kernel_hat) == 1:
            return self.dx * kernel_hat[:, np.newaxis] * values_hat
        return self.dx * kernel_hat * values_hat

    def _compute_smoothed_density_gradient(
        self,
        population,
        time,
        population_index=None,
    ):
        population_hat = self._fft_matrix(population)
        smoothed_hat = self._convolve_in_fourier_space(
            population_hat,
            time,
            population_index=population_index,
        )
        gradient_hat = self._apply_first_derivative_fourier(smoothed_hat)
        return self._ifft_matrix(gradient_hat)

    def _compute_smoothed_density_second_derivative(
        self,
        population,
        time,
        population_index=None,
    ):
        population_hat = self._fft_matrix(population)
        smoothed_hat = self._convolve_in_fourier_space(
            population_hat,
            time,
            population_index=population_index,
        )
        curvature_hat = self._apply_second_derivative_fourier(smoothed_hat)
        return self._ifft_matrix(curvature_hat)

    def _apply_interaction_matrix(self, values, time, population_index=None):
        effective_attraction = self._effective_attraction(time)
        if np.allclose(effective_attraction, 0.0):
            if population_index is None:
                return np.zeros_like(values)
            return np.zeros(values.shape[0], dtype=values.dtype)

        if population_index is None:
            return values @ effective_attraction.T

        return values @ effective_attraction[population_index, :]

    def _compute_advective_velocity(self, population, time):
        effective_attraction = self._effective_attraction(time)
        if np.allclose(effective_attraction, 0.0):
            return np.zeros_like(population)

        velocity = np.zeros_like(population)
        for population_index in range(self.number_of_population):
            smoothed_gradient = self._compute_smoothed_density_gradient(
                population,
                time,
                population_index=population_index,
            )
            velocity[:, population_index] = self._apply_interaction_matrix(
                smoothed_gradient,
                time,
                population_index=population_index,
            )

        return velocity

    def _compute_diffusion_term(self, population, time):
        effective_diffusion = self._effective_diffusion(time)
        if np.allclose(effective_diffusion, 0.0):
            return np.zeros_like(population)

        population_hat = self._fft_matrix(population)
        second_derivative_hat = self._apply_second_derivative_fourier(population_hat)
        second_derivative = self._ifft_matrix(second_derivative_hat)
        return second_derivative * effective_diffusion[np.newaxis, :]

    def _compute_flux_derivative(self, population, time):
        velocity = self._compute_advective_velocity(population, time)
        if np.allclose(velocity, 0.0):
            return np.zeros_like(population)

        flux = population * velocity
        flux_hat = self._fft_matrix(flux)
        derivative_hat = self._apply_first_derivative_fourier(flux_hat)
        return self._ifft_matrix(derivative_hat)

    def _compute_reaction_term(self, population, time):
        if self.reaction_term is None:
            return np.zeros_like(population)

        reaction_values = self.reaction_term(population, time, self)
        reaction_term = self._coerce_population_state(
            reaction_values,
            argument_name="reaction_term(population, time, model)",
        )
        if not np.all(np.isfinite(reaction_term)):
            raise ValueError("reaction_term must return finite values.")

        return reaction_term

    def _compute_rhs(self, time, population):
        diffusion_term = self._compute_diffusion_term(population, time)
        advection_term = self._compute_flux_derivative(population, time)
        reaction_term = self._compute_reaction_term(population, time)
        return diffusion_term - advection_term + reaction_term

    def _resolve_target_mass(self, candidate_population, reference_mass):
        if self.reaction_term is None:
            return np.asarray(reference_mass, dtype=float)

        target_mass = self.dx * np.sum(candidate_population, axis=0)
        if not np.all(np.isfinite(target_mass)):
            return None

        if np.any(target_mass < 0.0):
            return None

        return target_mass

    def _accept_step_candidate(self, candidate_population, reference_mass):
        if not np.all(np.isfinite(candidate_population)):
            return None

        target_mass = self._resolve_target_mass(candidate_population, reference_mass)
        if target_mass is None:
            return None

        max_population_by_species = np.max(
            np.abs(candidate_population),
            axis=0,
            keepdims=True,
        )
        negative_tolerance = np.maximum(
            1.0e-10,
            1.0e-5 * max_population_by_species,
        )
        if np.any(candidate_population < -negative_tolerance):
            return None

        clipped_population = np.maximum(candidate_population, 0.0)
        candidate_mass = self.dx * np.sum(clipped_population, axis=0)
        if not np.all(np.isfinite(candidate_mass)):
            return None

        accepted_population = np.zeros_like(clipped_population)
        positive_mass_mask = target_mass > 0.0
        if np.any(candidate_mass[positive_mass_mask] <= 0.0):
            return None

        if np.any(positive_mass_mask):
            mass_ratio = target_mass[positive_mass_mask] / candidate_mass[positive_mass_mask]
            accepted_population[:, positive_mass_mask] = (
                clipped_population[:, positive_mass_mask] * mass_ratio[np.newaxis, :]
            )

        return accepted_population

    def _runge_kutta_step(self, population, start_time, dt_step):
        k1 = self._compute_rhs(start_time, population)
        k2 = self._compute_rhs(
            start_time + 0.5 * dt_step,
            population + 0.5 * dt_step * k1,
        )
        k3 = self._compute_rhs(
            start_time + 0.5 * dt_step,
            population + 0.5 * dt_step * k2,
        )
        k4 = self._compute_rhs(
            start_time + dt_step,
            population + dt_step * k3,
        )
        return population + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _advance_one_output_step(self, population, start_time):
        next_population = np.asarray(population, dtype=float)
        current_time = start_time
        remaining_time = self.dt
        minimum_dt_step = self.dt * 1.0e-8

        while remaining_time > 0.0:
            dt_step = min(
                remaining_time,
                self._compute_internal_step_size(next_population, current_time),
            )
            reference_mass = self.dx * np.sum(next_population, axis=0)

            while True:
                candidate_population = self._runge_kutta_step(
                    next_population,
                    current_time,
                    dt_step,
                )
                accepted_population = self._accept_step_candidate(
                    candidate_population,
                    reference_mass,
                )
                if accepted_population is not None:
                    next_population = accepted_population
                    break

                dt_step *= 0.5
                if dt_step < minimum_dt_step:
                    raise ValueError(
                        "The simulation became non-finite; reduce dt or the interaction strength."
                    )

            current_time += dt_step
            remaining_time = max(remaining_time - dt_step, 0.0)

        return next_population

    def step(self, population, time):
        population = self._coerce_population_state(population)
        return self._advance_one_output_step(population, time)

    def solve(self):
        for step_index in range(self.number_of_steps):
            current_time = self.time[step_index]
            next_state = self._advance_one_output_step(
                self.U[step_index, :, :],
                current_time,
            )
            self.U[step_index + 1, :, :] = next_state
            self.U_fourier[step_index + 1, :, :] = self._fft_matrix(next_state)

        self._solution_computed = True
        return self.time, self.U

    def _ensure_solution_computed(self):
        if not self._solution_computed:
            self.solve()

    def get_solution(self):
        self._ensure_solution_computed()
        return self.U

    def get_fourier_solution(self):
        self._ensure_solution_computed()
        return self.U_fourier

    def get_snapshot(self, time_index):
        self._ensure_solution_computed()
        return self.U[time_index, :, :].copy()

    def get_final_distribution(self):
        self._ensure_solution_computed()
        return self.U[-1, :, :].copy()

    def get_final_initial_condition(self):
        final_distribution = self.get_final_distribution()
        return self._build_initial_condition_from_state(final_distribution)

    def get_mass(self):
        self._ensure_solution_computed()
        return self.dx * np.sum(self.U, axis=1)

    def _normalise_density_time_series(self, density_values):
        density_values = np.maximum(np.asarray(density_values, dtype=float), 0.0)
        masses = self.dx * np.sum(density_values, axis=1, keepdims=True)
        normalised_density = np.zeros_like(density_values)
        positive_mass_mask = masses[:, 0] > 0.0

        if np.any(positive_mass_mask):
            normalised_density[positive_mass_mask, :] = (
                density_values[positive_mass_mask, :] / masses[positive_mass_mask, :]
            )

        return normalised_density

    def get_overlap_energy(self, population_indices=(0, 1), observation_window=None):
        self._ensure_solution_computed()

        if self.number_of_population < 2:
            raise ValueError("get_overlap_energy requires at least two populations.")

        if observation_window is None:
            observation_window = self.cycle_period

        observation_window = float(observation_window)
        if observation_window <= 0.0:
            raise ValueError("observation_window must be positive.")

        try:
            population_a, population_b = population_indices
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "population_indices must contain exactly two population indices."
            ) from exc

        population_a = int(population_a)
        population_b = int(population_b)

        for population_index in (population_a, population_b):
            if not 0 <= population_index < self.number_of_population:
                raise IndexError("population index is out of bounds.")

        if population_a == population_b:
            raise ValueError("population_indices must refer to two distinct populations.")

        window_start = max(self.time[-1] - observation_window, self.time[0])
        window_mask = self.time >= (window_start - 1.0e-12)
        window_time = self.time[window_mask]

        density_a = self._normalise_density_time_series(
            self.U[window_mask, :, population_a]
        )
        density_b = self._normalise_density_time_series(
            self.U[window_mask, :, population_b]
        )
        overlap_density = np.sqrt(density_a * density_b)
        spatial_overlap = self.dx * np.sum(overlap_density, axis=1)
        return float(self._integrate_trapezoid(spatial_overlap, window_time))

    def get_kernel_components(self, time):
        smell_part, sight_part = self._blend_kernel_parts(
            self.smell_kernel_values,
            self.sight_kernel_values,
            time,
        )
        combined = smell_part + sight_part
        return {
            "grid": self.kernel_grid.copy(),
            "smell": smell_part.copy(),
            "sight": sight_part.copy(),
            "combined": combined.copy(),
        }

    def get_effective_parameters(self, time):
        combined_kernel = self._combined_kernel_values(time)
        activity_mask = self.get_activity_mask(time)
        if np.ndim(combined_kernel) == 2:
            kernel_mass = self.dx * np.sum(combined_kernel, axis=0)
        else:
            kernel_mass = float(self.dx * np.sum(combined_kernel))
        return {
            "is_daytime": self.is_daytime(time),
            "is_active": bool(np.any(activity_mask)),
            "all_populations_active": bool(np.all(activity_mask)),
            "active_populations": activity_mask.copy(),
            "activity_factor": activity_mask.astype(float),
            "coefficient_diffusion": self._effective_diffusion(time).copy(),
            "coefficient_attraction": self._effective_attraction(time).copy(),
            "kernel_mass": kernel_mass,
        }

    def _build_initial_condition_from_state(self, state):
        state = self._coerce_population_state(state, argument_name="state")

        x_extended = np.concatenate((self.x, [self.b_border]))
        state_extended = np.vstack((state, state[0:1, :]))

        def initial_condition(x):
            sample_points = np.asarray(x, dtype=float)
            wrapped_points = self.a_border + np.mod(
                sample_points - self.a_border,
                self.length,
            )
            interpolated_profiles = [
                np.interp(
                    wrapped_points,
                    x_extended,
                    state_extended[:, population_index],
                )
                for population_index in range(self.number_of_population)
            ]
            if self.number_of_population == 1:
                return interpolated_profiles[0]
            return np.column_stack(interpolated_profiles)

        return initial_condition

    def _resolve_output_path(self, save_path, default_filename):
        output_path = Path(save_path) if save_path is not None else Path(default_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _compute_figure_layout(self, number_of_plots):
        columns = int(np.ceil(np.sqrt(number_of_plots)))
        rows = int(np.ceil(number_of_plots / columns))
        return rows, columns

    def _select_snapshot_indices(self, number_of_plots):
        available_snapshots = self.number_of_steps + 1
        if number_of_plots < 1:
            raise ValueError("number_of_plots must be at least 1.")
        if number_of_plots > available_snapshots:
            raise ValueError(
                "number_of_plots cannot exceed the number of stored time snapshots."
            )

        indices = np.linspace(0, available_snapshots - 1, number_of_plots)
        return np.round(indices).astype(int)

    def _compute_solution_ylim(self):
        solution_min = float(np.min(self.U))
        solution_max = float(np.max(self.U))
        amplitude = solution_max - solution_min
        margin = 0.05 * amplitude if amplitude > 0.0 else 0.1 * max(solution_max, 1.0)
        return solution_min - margin, solution_max + margin

    def _periodic_event_times(self, reference_time):
        first_index = int(np.ceil((self.time[0] - reference_time) / self.cycle_period))
        last_index = int(np.floor((self.time[-1] - reference_time) / self.cycle_period))
        return [
            reference_time + cycle_index * self.cycle_period
            for cycle_index in range(first_index, last_index + 1)
        ]

    def _day_to_night_switch_times(self):
        return self._periodic_event_times(self.day_start + self.day_duration)

    def _night_to_day_switch_times(self):
        return self._periodic_event_times(self.day_start)

    def _population_activity_switch_times(self, population_index):
        start_times = []
        end_times = []
        for interval_start, interval_duration in self.activity_intervals[population_index]:
            if interval_duration >= self.cycle_period:
                continue

            start_times.extend(self._periodic_event_times(interval_start))
            end_times.extend(
                self._periodic_event_times(interval_start + interval_duration)
            )

        if not start_times and not end_times:
            return {"start": [], "end": []}

        return {
            "start": np.unique(np.round(start_times, decimals=12)).tolist(),
            "end": np.unique(np.round(end_times, decimals=12)).tolist(),
        }

    def _plot_population_profiles(self, axis, state, title):
        for population_index in range(self.number_of_population):
            axis.plot(
                self.x,
                state[:, population_index],
                label=f"Population {population_index + 1}",
            )

        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("density")

    def plot_solution_snapshots(self, number_of_plots=4, save=False, save_path=None):
        self._ensure_solution_computed()

        snapshot_indices = self._select_snapshot_indices(number_of_plots)
        rows, columns = self._compute_figure_layout(number_of_plots)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.5 * columns, 3.5 * rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = axes.ravel()
        y_min, y_max = self._compute_solution_ylim()

        for axis, snapshot_index in zip(flat_axes, snapshot_indices):
            title = f"t = {self.time[snapshot_index]:.3f}"
            self._plot_population_profiles(axis, self.U[snapshot_index, :, :], title)
            axis.set_xlim(self.a_border, self.b_border)
            axis.set_ylim(y_min, y_max)

        for axis in flat_axes[len(snapshot_indices) :]:
            axis.set_visible(False)

        if snapshot_indices.size > 0:
            flat_axes[0].legend(loc="upper right")

        figure.tight_layout()

        if save:
            output_path = self._resolve_output_path(save_path, "solution_snapshots.png")
            figure.savefig(output_path, bbox_inches="tight")

        return figure, axes

    def create_solution_gif(
        self,
        interval=100,
        save=False,
        save_path=None,
        fps=15,
    ):
        self._ensure_solution_computed()

        figure, axis = plt.subplots(figsize=(8, 4.5))
        y_min, y_max = self._compute_solution_ylim()
        lines = [
            axis.plot([], [], label=f"Population {population_index + 1}")[0]
            for population_index in range(self.number_of_population)
        ]

        axis.set_xlim(self.a_border, self.b_border)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("density")
        axis.legend(loc="upper right")

        def update(frame_index):
            state = self.U[frame_index, :, :]
            for population_index, line in enumerate(lines):
                line.set_data(self.x, state[:, population_index])

            axis.set_title(f"Solution at t = {self.time[frame_index]:.3f}")
            return lines

        animation = FuncAnimation(
            figure,
            update,
            frames=self.number_of_steps + 1,
            interval=interval,
            blit=False,
        )

        if save:
            output_path = self._resolve_output_path(save_path, "solution.gif")
            animation.save(output_path, writer=PillowWriter(fps=fps))

        return figure, animation

    def _add_transition_markers(
        self,
        axis,
        population_index,
        show_legend,
        show_day_night_cycle=True,
        show_activity_period=True,
    ):
        if show_day_night_cycle:
            for switch_index, switch_time in enumerate(self._night_to_day_switch_times()):
                label = "night to day" if show_legend and switch_index == 0 else None
                axis.axhline(
                    switch_time,
                    color="blue",
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.9,
                    label=label,
                )

            for switch_index, switch_time in enumerate(self._day_to_night_switch_times()):
                label = "day to night" if show_legend and switch_index == 0 else None
                axis.axhline(
                    switch_time,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=label,
                )

        if show_activity_period:
            activity_switches = self._population_activity_switch_times(population_index)
            for switch_index, switch_time in enumerate(activity_switches["start"]):
                label = "activity starts" if show_legend and switch_index == 0 else None
                axis.axhline(
                    switch_time,
                    color="orange",
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                )

            for switch_index, switch_time in enumerate(activity_switches["end"]):
                label = "activity ends" if show_legend and switch_index == 0 else None
                axis.axhline(
                    switch_time,
                    color="orange",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    label=label,
                )

        if show_legend:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(loc="upper right")

    def plot_solution_heatmaps(
        self,
        cmap="hot_r",
        share_color_scale=True,
        show_day_night_cycle=True,
        show_activity_period=True,
        save=False,
        save_path=None,
    ):
        self._ensure_solution_computed()

        figure, axes = plt.subplots(
            1,
            self.number_of_population,
            figsize=(4.8 * self.number_of_population, 4.0),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = axes.ravel()

        common_limits = {}
        if share_color_scale:
            common_limits = {
                "vmin": float(np.min(self.U)),
                "vmax": float(np.max(self.U)),
            }

        extent = [self.a_border, self.b_border, self.time[0], self.time[-1]]

        for population_index, axis in enumerate(flat_axes):
            heatmap_values = self.U[:, :, population_index]
            image = axis.imshow(
                heatmap_values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                **common_limits,
            )
            axis.set_title(f"Population {population_index + 1}")
            axis.set_xlabel("x")
            if population_index == 0:
                axis.set_ylabel("t")
            self._add_transition_markers(
                axis,
                population_index,
                show_legend=population_index == 0,
                show_day_night_cycle=show_day_night_cycle,
                show_activity_period=show_activity_period,
            )
            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("density")

        figure.tight_layout()

        if save:
            output_path = self._resolve_output_path(save_path, "solution_heatmaps.png")
            figure.savefig(output_path, bbox_inches="tight")

        return figure, axes

    def plot_solution_heatmap(
        self,
        cmap="hot_r",
        share_color_scale=True,
        show_day_night_cycle=True,
        show_activity_period=True,
        save=False,
        save_path=None,
    ):
        figure, axes = self.plot_solution_heatmaps(
            cmap=cmap,
            share_color_scale=share_color_scale,
            show_day_night_cycle=show_day_night_cycle,
            show_activity_period=show_activity_period,
            save=save,
            save_path=save_path,
        )

        if self.number_of_population == 1:
            axis = axes.ravel()[0]
            axis.set_title("One-population day-night simulation")
            return figure, axis

        return figure, axes
