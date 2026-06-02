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
        activity_mode="always",
        activity_start=None,
        activity_end=None,
        sight_weight=0.5,
        sight_radius=0.05,
        smell_radius=0.15,
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
        self.day_start = float(day_start)
        self.day_end = (
            0.5 * self.cycle_period if day_end is None else float(day_end)
        )
        self.activity_mode = str(activity_mode)
        self._has_explicit_activity_period = (
            activity_start is not None or activity_end is not None
        )
        self.activity_start, self.activity_end = self._resolve_activity_interval(
            activity_start,
            activity_end,
        )
        self._always_active = (
            not self._has_explicit_activity_period and self.activity_mode == "always"
        )
        self.sight_weight = float(sight_weight)
        self.sight_radius = float(sight_radius)
        self.smell_radius = float(smell_radius)

        self.length = self._compute_domain_length()
        self.number_of_steps = self._compute_number_of_steps()
        self.day_duration = self._compute_interval_duration(
            self.day_start,
            self.day_end,
        )
        self.activity_duration = self._compute_activity_duration()

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

    def _compute_domain_length(self):
        return self.b_border - self.a_border

    def _compute_number_of_steps(self):
        return int(round(self.total_time / self.dt))

    def _resolve_activity_interval(self, activity_start, activity_end):
        if (activity_start is None) != (activity_end is None):
            raise ValueError(
                "Provide both activity_start and activity_end, or neither."
            )

        if activity_start is not None:
            return float(activity_start), float(activity_end)

        if self.activity_mode == "diurnal":
            return self.day_start, self.day_end

        if self.activity_mode == "nocturnal":
            return self.day_end, self.day_start

        return self.day_start, self.day_start + self.cycle_period

    def _compute_interval_duration(self, start, end, allow_full_cycle=False):
        if not np.isfinite(self.cycle_period) or self.cycle_period <= 0.0:
            return np.nan

        raw_duration = float(end) - float(start)
        wrapped_duration = np.mod(raw_duration, self.cycle_period)

        if allow_full_cycle and np.isclose(wrapped_duration, 0.0):
            cycle_count = raw_duration / self.cycle_period
            if not np.isclose(cycle_count, 0.0) and np.isclose(
                cycle_count,
                round(cycle_count),
            ):
                return self.cycle_period

        return wrapped_duration

    def _compute_activity_duration(self):
        if self._always_active:
            return self.cycle_period

        return self._compute_interval_duration(
            self.activity_start,
            self.activity_end,
            allow_full_cycle=True,
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

        if not np.isfinite(self.day_start) or not np.isfinite(self.day_end):
            raise ValueError("day_start and day_end must be finite.")

        if np.isclose(self.day_duration, 0.0):
            raise ValueError(
                "day_start and day_end must define non-zero day and night intervals."
            )

        if self.activity_mode not in {"always", "diurnal", "nocturnal"}:
            raise ValueError(
                "activity_mode must be 'always', 'diurnal', or 'nocturnal'."
            )

        if not np.isfinite(self.activity_start) or not np.isfinite(self.activity_end):
            raise ValueError("activity_start and activity_end must be finite.")

        if not self._always_active and np.isclose(self.activity_duration, 0.0):
            raise ValueError(
                "activity_start and activity_end must define a non-zero active interval."
            )

        if not 0.0 <= self.sight_weight <= 1.0:
            raise ValueError("sight_weight must lie in the interval [0, 1].")

        if not np.isfinite(self.sight_radius) or self.sight_radius <= 0.0:
            raise ValueError("sight_radius must be positive.")

        if not np.isfinite(self.smell_radius) or self.smell_radius <= 0.0:
            raise ValueError("smell_radius must be positive.")

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

        smoothed_curvature = self._compute_smoothed_density_second_derivative(
            population,
            time,
        )
        compression_rate = self._apply_interaction_matrix(smoothed_curvature, time)
        max_compression_rate = float(np.max(np.abs(compression_rate)))
        if max_compression_rate <= 0.0 or not np.isfinite(max_compression_rate):
            return np.inf

        safety_factor = 0.25
        return safety_factor / max_compression_rate

    def _compute_positivity_dt_limit(self, population, time):
        rhs = self._compute_rhs(time, population)
        unstable_mask = (rhs < 0.0) & (population > 0.0)
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

    def _smell_kernel(self, values):
        values = np.asarray(values, dtype=float)
        denominator = self.smell_radius * np.sqrt(2.0 * np.pi)
        return np.exp(-0.5 * (values / self.smell_radius) ** 2) / denominator

    def _sight_kernel(self, values):
        values = np.asarray(values, dtype=float)
        return 0.5 * np.exp(-np.abs(values) / self.sight_radius) / self.sight_radius

    def _normalise_kernel_values(self, kernel_values):
        normalisation = self.dx * np.sum(kernel_values)
        if normalisation <= 0.0:
            raise ValueError("Kernel must have positive mass.")
        return kernel_values / normalisation

    def _generate_smell_kernel_values(self):
        return self._normalise_kernel_values(self._smell_kernel(self.kernel_grid))

    def _generate_sight_kernel_values(self):
        return self._normalise_kernel_values(self._sight_kernel(self.kernel_grid))

    def _fft_vector(self, values):
        return np.fft.fft(values)

    def _ifft_vector(self, values):
        return np.fft.ifft(values).real

    def _fft_matrix(self, values):
        return np.fft.fft(values, axis=0)

    def _ifft_matrix(self, values):
        return np.fft.ifft(values, axis=0).real

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

    def is_active(self, time):
        if self._always_active:
            return True
        return self._is_within_interval(time, self.activity_start, self.activity_duration)

    def _activity_factor(self, time):
        return 1.0 if self.is_active(time) else 0.0

    def _effective_diffusion(self, time):
        return self.coefficient_diffusion * self._activity_factor(time)

    def _effective_attraction(self, time):
        return self.coefficient_attraction * self._activity_factor(time)

    def _sight_factor(self, time):
        return 1.0 if self.is_daytime(time) else 0.0

    def _combined_kernel_values(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_values
        sight_part = (
            self.sight_weight * self._sight_factor(time) * self.sight_kernel_values
        )
        return smell_part + sight_part

    def _combined_kernel_fourier(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_fourier
        sight_part = (
            self.sight_weight * self._sight_factor(time) * self.sight_kernel_fourier
        )
        return smell_part + sight_part

    def _convolve_in_fourier_space(self, values_hat, time):
        kernel_hat = self._combined_kernel_fourier(time)[:, np.newaxis]
        return self.dx * kernel_hat * values_hat

    def _compute_smoothed_density_gradient(self, population, time):
        population_hat = self._fft_matrix(population)
        smoothed_hat = self._convolve_in_fourier_space(population_hat, time)
        gradient_hat = self._apply_first_derivative_fourier(smoothed_hat)
        return self._ifft_matrix(gradient_hat)

    def _compute_smoothed_density_second_derivative(self, population, time):
        population_hat = self._fft_matrix(population)
        smoothed_hat = self._convolve_in_fourier_space(population_hat, time)
        curvature_hat = self._apply_second_derivative_fourier(smoothed_hat)
        return self._ifft_matrix(curvature_hat)

    def _apply_interaction_matrix(self, values, time):
        effective_attraction = self._effective_attraction(time)
        if np.allclose(effective_attraction, 0.0):
            return np.zeros_like(values)
        return values @ effective_attraction.T

    def _compute_advective_velocity(self, population, time):
        smoothed_gradient = self._compute_smoothed_density_gradient(population, time)
        return self._apply_interaction_matrix(smoothed_gradient, time)

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

    def _compute_rhs(self, time, population):
        diffusion_term = self._compute_diffusion_term(population, time)
        advection_term = self._compute_flux_derivative(population, time)
        return diffusion_term - advection_term

    def _accept_step_candidate(self, candidate_population, reference_mass):
        if not np.all(np.isfinite(candidate_population)):
            return None

        negative_tolerance = 1.0e-10
        if float(np.min(candidate_population)) < -negative_tolerance:
            return None

        clipped_population = np.maximum(candidate_population, 0.0)
        candidate_mass = self.dx * np.sum(clipped_population, axis=0)
        if np.any(candidate_mass <= 0.0) or not np.all(np.isfinite(candidate_mass)):
            return None

        mass_ratio = reference_mass / candidate_mass
        return clipped_population * mass_ratio[np.newaxis, :]

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

    def get_kernel_components(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_values
        sight_part = (
            self.sight_weight * self._sight_factor(time) * self.sight_kernel_values
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
        return {
            "is_daytime": self.is_daytime(time),
            "is_active": self.is_active(time),
            "activity_factor": self._activity_factor(time),
            "coefficient_diffusion": self._effective_diffusion(time).copy(),
            "coefficient_attraction": self._effective_attraction(time).copy(),
            "kernel_mass": self.dx * np.sum(combined_kernel),
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

    def _activity_switch_times(self):
        if not self._has_explicit_activity_period or self.activity_duration >= self.cycle_period:
            return {"start": [], "end": []}

        return {
            "start": self._periodic_event_times(self.activity_start),
            "end": self._periodic_event_times(self.activity_start + self.activity_duration),
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

    def _add_transition_markers(self, axis, show_legend):
        for switch_index, switch_time in enumerate(self._day_to_night_switch_times()):
            label = "day to night" if show_legend and switch_index == 0 else None
            axis.axhline(
                switch_time,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=label,
            )

        activity_switches = self._activity_switch_times()
        for switch_index, switch_time in enumerate(activity_switches["start"]):
            label = "activity starts" if show_legend and switch_index == 0 else None
            axis.axhline(
                switch_time,
                color="white",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
                label=label,
            )

        for switch_index, switch_time in enumerate(activity_switches["end"]):
            label = "activity ends" if show_legend and switch_index == 0 else None
            axis.axhline(
                switch_time,
                color="white",
                linestyle="-.",
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
            self._add_transition_markers(axis, show_legend=population_index == 0)
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
        save=False,
        save_path=None,
    ):
        figure, axes = self.plot_solution_heatmaps(
            cmap=cmap,
            share_color_scale=share_color_scale,
            save=save,
            save_path=save_path,
        )

        if self.number_of_population == 1:
            axis = axes.ravel()[0]
            axis.set_title("One-population day-night simulation")
            return figure, axis

        return figure, axes
