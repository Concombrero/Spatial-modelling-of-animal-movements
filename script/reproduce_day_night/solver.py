import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class DayNightModel1D:
    """One-dimensional spectral solver for a single-population day-night model."""

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
        day_start=0.0,
        day_end=None,
        activity_mode="always",
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
        self.coefficient_attraction = float(coefficient_attraction)
        self.coefficient_diffusion = float(coefficient_diffusion)
        self.cycle_period = float(cycle_period)
        self.day_start = float(day_start)
        self.day_end = (
            0.5 * self.cycle_period if day_end is None else float(day_end)
        )
        self.activity_mode = str(activity_mode)
        self.sight_weight = float(sight_weight)
        self.sight_radius = float(sight_radius)
        self.smell_radius = float(smell_radius)

        self.length = self._compute_domain_length()
        self.number_of_steps = self._compute_number_of_steps()
        self.day_duration = self._compute_day_duration()

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
        self.U[0, :] = self._evaluate_initial_condition()

        self.U_fourier = np.zeros_like(self.U, dtype=complex)
        self.U_fourier[0, :] = self._fft_vector(self.U[0, :])
        self._solution_computed = False

    def _compute_domain_length(self):
        return self.b_border - self.a_border

    def _compute_number_of_steps(self):
        return int(round(self.total_time / self.dt))

    def _compute_day_duration(self):
        if not np.isfinite(self.cycle_period) or self.cycle_period <= 0.0:
            return np.nan

        return np.mod(self.day_end - self.day_start, self.cycle_period)

    def _validate_inputs(self):
        if self.length <= 0.0:
            raise ValueError("b_border must be larger than a_border.")

        if self.number_of_points < 2:
            raise ValueError("number_of_points must be at least 2.")

        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")

        if self.total_time < 0.0:
            raise ValueError("total_time must be non-negative.")

        if not np.isclose(self.number_of_steps * self.dt, self.total_time):
            raise ValueError("total_time must be an integer multiple of dt.")

        if not np.isfinite(self.coefficient_attraction):
            raise ValueError("coefficient_attraction must be finite.")

        if not np.isfinite(self.coefficient_diffusion):
            raise ValueError("coefficient_diffusion must be finite.")

        if self.coefficient_diffusion < 0.0:
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
        max_wavenumber_squared = float(np.max(self.wavenumbers**2))
        stability_scale = self.dt * self.coefficient_diffusion * max_wavenumber_squared
        stable_limit = 2.5

        if stability_scale <= stable_limit:
            return 1

        return int(np.ceil(stability_scale / stable_limit))

    def _compute_advection_dt_limit(self, population, time):
        effective_attraction = self._effective_attraction(time)
        if effective_attraction == 0.0:
            return np.inf

        smoothed_gradient = self._compute_smoothed_density_gradient(population, time)
        max_velocity = float(np.max(np.abs(effective_attraction * smoothed_gradient)))
        if max_velocity <= 0.0 or not np.isfinite(max_velocity):
            return np.inf

        cfl_number = 0.5
        return cfl_number * self.dx / max_velocity

    def _compute_compression_dt_limit(self, population, time):
        effective_attraction = self._effective_attraction(time)
        if effective_attraction == 0.0:
            return np.inf

        smoothed_curvature = self._compute_smoothed_density_second_derivative(
            population,
            time,
        )
        max_compression_rate = float(
            np.max(np.abs(effective_attraction * smoothed_curvature))
        )
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
        return np.zeros((self.number_of_steps + 1, self.number_of_points), dtype=float)

    def _evaluate_initial_condition(self):
        if self.initial_condition is None:
            return self._build_default_initial_condition()

        values = np.asarray(self.initial_condition(self.x), dtype=float)

        if values.shape == (self.number_of_points,):
            return self._normalise_profile(values)

        if values.shape == (1, self.number_of_points):
            return self._normalise_profile(values[0, :])

        if values.shape == (self.number_of_points, 1):
            return self._normalise_profile(values[:, 0])

        raise ValueError(
            "initial_condition(x) must return an array of shape "
            "(number_of_points,), (1, number_of_points), or (number_of_points, 1)."
        )

    def _build_default_initial_condition(self):
        center = self.a_border + 0.35 * self.length
        width = max(self.length / 12.0, self.dx)
        return self._normalise_profile(self._build_periodic_gaussian(center, width))

    def _build_periodic_gaussian(self, center, width):
        wrapped_distance = (
            (self.x - center + 0.5 * self.length) % self.length
        ) - 0.5 * self.length
        return np.exp(-0.5 * (wrapped_distance / width) ** 2)

    def _normalise_profile(self, values):
        values = np.asarray(values, dtype=float)
        mass = self.dx * np.sum(values)
        if mass <= 0.0:
            raise ValueError("initial_condition must have positive mass.")
        return values / mass

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

    def _first_derivative_multiplier(self):
        return 1j * self.wavenumbers

    def _second_derivative_multiplier(self):
        return -(self.wavenumbers**2)

    def _phase_in_cycle(self, time):
        return np.mod(time - self.day_start, self.cycle_period)

    def is_daytime(self, time):
        return bool(self._phase_in_cycle(time) < self.day_duration)

    def _activity_factor(self, time):
        if self.activity_mode == "always":
            return 1.0

        is_daytime = self.is_daytime(time)
        if self.activity_mode == "diurnal":
            return 1.0 if is_daytime else 0.0

        return 0.0 if is_daytime else 1.0

    def _effective_diffusion(self, time):
        return self.coefficient_diffusion * self._activity_factor(time)

    def _effective_attraction(self, time):
        return self.coefficient_attraction * self._activity_factor(time)

    def _sight_factor(self, time):
        return 1.0 if self.is_daytime(time) else 0.0

    def _combined_kernel_values(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_values
        sight_part = self.sight_weight * self._sight_factor(time) * self.sight_kernel_values
        return smell_part + sight_part

    def _combined_kernel_fourier(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_fourier
        sight_part = self.sight_weight * self._sight_factor(time) * self.sight_kernel_fourier
        return smell_part + sight_part

    def _compute_smoothed_density_gradient(self, population, time):
        population_hat = self._fft_vector(population)
        smoothed_hat = self.dx * self._combined_kernel_fourier(time) * population_hat
        gradient_hat = self._first_derivative_multiplier() * smoothed_hat
        return self._ifft_vector(gradient_hat)

    def _compute_smoothed_density_second_derivative(self, population, time):
        population_hat = self._fft_vector(population)
        smoothed_hat = self.dx * self._combined_kernel_fourier(time) * population_hat
        curvature_hat = self._second_derivative_multiplier() * smoothed_hat
        return self._ifft_vector(curvature_hat)

    def _compute_diffusion_term(self, population, time):
        effective_diffusion = self._effective_diffusion(time)
        if effective_diffusion == 0.0:
            return np.zeros_like(population)

        population_hat = self._fft_vector(population)
        second_derivative_hat = self._second_derivative_multiplier() * population_hat
        return effective_diffusion * self._ifft_vector(second_derivative_hat)

    def _compute_flux_derivative(self, population, time):
        effective_attraction = self._effective_attraction(time)
        if effective_attraction == 0.0:
            return np.zeros_like(population)

        smoothed_gradient = self._compute_smoothed_density_gradient(population, time)
        flux = population * (effective_attraction * smoothed_gradient)
        flux_hat = self._fft_vector(flux)
        derivative_hat = self._first_derivative_multiplier() * flux_hat
        return self._ifft_vector(derivative_hat)

    def _compute_rhs(self, time, population):
        diffusion_term = self._compute_diffusion_term(population, time)
        advection_term = self._compute_flux_derivative(population, time)
        return diffusion_term - advection_term

    def _accept_step_candidate(self, candidate_population, reference_mass):
        if not np.all(np.isfinite(candidate_population)):
            return None

        negative_tolerance = 1.0e-10
        if np.min(candidate_population) < -negative_tolerance:
            return None

        clipped_population = np.maximum(candidate_population, 0.0)
        candidate_mass = self.dx * np.sum(clipped_population)
        if candidate_mass <= 0.0 or not np.isfinite(candidate_mass):
            return None

        return clipped_population * (reference_mass / candidate_mass)

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
        next_population = population
        current_time = start_time
        remaining_time = self.dt
        minimum_dt_step = self.dt * 1.0e-8

        while remaining_time > 0.0:
            dt_step = min(
                remaining_time,
                self._compute_internal_step_size(next_population, current_time),
            )
            reference_mass = self.dx * np.sum(next_population)

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
            remaining_time -= dt_step

        return next_population

    def step(self, population, time):
        return self._advance_one_output_step(population, time)

    def solve(self):
        for step_index in range(self.number_of_steps):
            current_time = self.time[step_index]
            next_state = self._advance_one_output_step(self.U[step_index, :], current_time)
            self.U[step_index + 1, :] = next_state
            self.U_fourier[step_index + 1, :] = self._fft_vector(next_state)

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
        return self.U[time_index, :].copy()

    def get_final_distribution(self):
        self._ensure_solution_computed()
        return self.U[-1, :].copy()

    def get_final_initial_condition(self):
        final_distribution = self.get_final_distribution()
        return self._build_initial_condition_from_state(final_distribution)

    def get_mass(self):
        self._ensure_solution_computed()
        return self.dx * np.sum(self.U, axis=1)

    def get_kernel_components(self, time):
        smell_part = (1.0 - self.sight_weight) * self.smell_kernel_values
        sight_part = self.sight_weight * self._sight_factor(time) * self.sight_kernel_values
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
            "activity_factor": self._activity_factor(time),
            "coefficient_diffusion": self._effective_diffusion(time),
            "coefficient_attraction": self._effective_attraction(time),
            "kernel_mass": self.dx * np.sum(combined_kernel),
        }

    def _build_initial_condition_from_state(self, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (self.number_of_points,):
            raise ValueError("state must have shape (number_of_points,).")

        x_extended = np.concatenate((self.x, [self.b_border]))
        state_extended = np.concatenate((state, [state[0]]))

        def initial_condition(x):
            sample_points = np.asarray(x, dtype=float)
            wrapped_points = self.a_border + np.mod(
                sample_points - self.a_border,
                self.length,
            )
            return np.interp(wrapped_points, x_extended, state_extended)

        return initial_condition

    def _resolve_output_path(self, save_path, default_filename):
        output_path = Path(save_path) if save_path is not None else Path(default_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

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

    def _day_to_night_switch_times(self):
        first_switch_time = self.day_start + self.day_duration
        number_of_cycles = int(np.floor((self.time[-1] - first_switch_time) / self.cycle_period))

        switch_times = []
        for cycle_index in range(number_of_cycles + 1):
            switch_time = first_switch_time + cycle_index * self.cycle_period
            if self.time[0] <= switch_time <= self.time[-1]:
                switch_times.append(switch_time)

        return switch_times

    def plot_solution_snapshots(self, number_of_plots=4, save=False, save_path=None):
        self._ensure_solution_computed()

        snapshot_indices = self._select_snapshot_indices(number_of_plots)
        figure, axes = plt.subplots(
            number_of_plots,
            1,
            figsize=(7.0, 2.7 * number_of_plots),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        y_min, y_max = self._compute_solution_ylim()

        for axis, snapshot_index in zip(flat_axes, snapshot_indices):
            axis.plot(self.x, self.U[snapshot_index, :], color="black")
            axis.set_title(f"t = {self.time[snapshot_index]:.3f}")
            axis.set_ylabel("density")
            axis.set_xlim(self.a_border, self.b_border)
            axis.set_ylim(y_min, y_max)

        flat_axes[-1].set_xlabel("x")
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
        line = axis.plot([], [], color="black", label="Population")[0]

        axis.set_xlim(self.a_border, self.b_border)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("x")
        axis.set_ylabel("density")
        axis.legend(loc="upper right")

        def update(frame_index):
            line.set_data(self.x, self.U[frame_index, :])
            axis.set_title(f"Solution at t = {self.time[frame_index]:.3f}")
            return (line,)

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

    def plot_solution_heatmap(
        self,
        cmap="hot_r",
        save=False,
        save_path=None,
    ):
        self._ensure_solution_computed()

        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        image = axis.imshow(
            self.U,
            origin="lower",
            aspect="auto",
            extent=[self.a_border, self.b_border, self.time[0], self.time[-1]],
            cmap=cmap,
        )
        axis.set_title("One-population day-night simulation")
        axis.set_xlabel("x")
        axis.set_ylabel("t")

        switch_times = self._day_to_night_switch_times()
        for switch_index, switch_time in enumerate(switch_times):
            label = "day to night" if switch_index == 0 else None
            axis.axhline(
                switch_time,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label=label,
            )

        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("density")
        if switch_times:
            axis.legend(loc="upper right")
        figure.tight_layout()

        if save:
            output_path = self._resolve_output_path(save_path, "solution_heatmap.png")
            figure.savefig(output_path, bbox_inches="tight")

        return figure, axis
