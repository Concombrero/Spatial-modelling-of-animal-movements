import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import i0e
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Model1D:
    """One-dimensional spectral solver for the non-local advection-diffusion model."""

    def __init__(
        self,
        a_border,
        b_border,
        number_of_points,
        total_time,
        dt,
        initial_condition=None,
        *,
        number_of_population,
        coefficient_attraction,
        coefficient_diffusion,
        kernel_type,
        kernel_coefficient=None,
        kernel_standard_deviation=None,
    ):
        self.a_border = float(a_border)
        self.b_border = float(b_border)
        self.number_of_points = int(number_of_points)
        self.total_time = float(total_time)
        self.dt = float(dt)
        self.initial_condition = initial_condition
        self.number_of_population = int(number_of_population)
        self.coefficient_attraction = np.asarray(coefficient_attraction, dtype=float)
        self.coefficient_diffusion = np.asarray(coefficient_diffusion, dtype=float)
        self.kernel_type = kernel_type

        self.length = self._compute_domain_length()
        self.number_of_steps = self._compute_number_of_steps()
        self.kernel_coefficient = self._resolve_kernel_coefficient(
            kernel_coefficient,
            kernel_standard_deviation,
        )

        self._validate_inputs()

        self.dx = self._compute_dx()
        self.x = self._build_space_grid()
        self.time = self._build_time_grid()
        self.time_steps = self.number_of_steps
        self.wavenumbers = self._build_wavenumbers()
        self.rk4_substeps = self._compute_rk4_substeps()
        self.internal_dt = self.dt / self.rk4_substeps

        self.kernel_values = self._generate_kernel_values()
        self.kernel_fourier = self._fft_vector(self.kernel_values)

        self.U = self._initialise_solution_storage()
        self.U[0, :, :] = self._evaluate_initial_condition()

        self.U_fourier = np.zeros_like(self.U, dtype=complex)
        self.U_fourier[0, :, :] = self._fft_matrix(self.U[0, :, :])
        self._solution_computed = False

    def _compute_domain_length(self):
        """Return the size of the periodic spatial domain."""
        return self.b_border - self.a_border

    def _compute_number_of_steps(self):
        """Return the number of time steps implied by total_time and dt."""
        return int(round(self.total_time / self.dt))

    def _validate_inputs(self):
        """Check that grid, time, and coefficient inputs are consistent."""
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

        if self.coefficient_attraction.shape != (
            self.number_of_population,
            self.number_of_population,
        ):
            raise ValueError(
                "coefficient_attraction must have shape "
                "(number_of_population, number_of_population)."
            )

        if self.coefficient_diffusion.shape != (self.number_of_population,):
            raise ValueError(
                "coefficient_diffusion must have length number_of_population."
            )

        if np.any(self.coefficient_diffusion < 0.0):
            raise ValueError("Diffusion coefficients must be non-negative.")

        self._validate_kernel_inputs()

    def _resolve_kernel_coefficient(
        self,
        kernel_coefficient,
        kernel_standard_deviation,
    ):
        """Resolve the kernel parameter from either a direct coefficient or a target width."""
        has_coefficient = kernel_coefficient is not None
        has_standard_deviation = kernel_standard_deviation is not None

        if has_coefficient == has_standard_deviation:
            raise ValueError(
                "Provide exactly one of kernel_coefficient or "
                "kernel_standard_deviation."
            )

        if has_coefficient:
            return float(kernel_coefficient)

        standard_deviation = float(kernel_standard_deviation)
        return self._compute_kernel_coefficient_from_standard_deviation(
            standard_deviation
        )

    def _maximum_kernel_standard_deviation(self):
        """Return the largest attainable kernel standard deviation on the domain."""
        return self.length / (2.0 * np.sqrt(3.0))

    def _compute_kernel_coefficient_from_standard_deviation(self, standard_deviation):
        """Convert a target kernel standard deviation into the kernel parameter."""
        if not np.isfinite(standard_deviation):
            raise ValueError("kernel_standard_deviation must be finite.")

        if standard_deviation <= 0.0:
            raise ValueError("kernel_standard_deviation must be positive.")

        maximum_standard_deviation = self._maximum_kernel_standard_deviation()
        if (
            standard_deviation > maximum_standard_deviation
            and not np.isclose(standard_deviation, maximum_standard_deviation)
        ):
            raise ValueError(
                "kernel_standard_deviation must be at most "
                "(b_border - a_border) / (2 * sqrt(3))."
            )

        if self.kernel_type == "top_hat":
            return np.sqrt(3.0) * standard_deviation

        if self.kernel_type == "von_mises":
            return self._compute_von_mises_coefficient_from_standard_deviation(
                standard_deviation
            )

        raise ValueError("Unsupported kernel type.")

    def _compute_von_mises_coefficient_from_standard_deviation(
        self,
        standard_deviation,
    ):
        """Match a von Mises concentration to the requested kernel width."""
        maximum_standard_deviation = self._compute_kernel_standard_deviation_from_coefficient(
            0.0
        )
        if np.isclose(standard_deviation, maximum_standard_deviation):
            return 0.0

        def objective(kernel_coefficient):
            return (
                self._compute_kernel_standard_deviation_from_coefficient(
                    kernel_coefficient
                )
                - standard_deviation
            )

        upper_bound = 1.0
        while objective(upper_bound) > 0.0:
            upper_bound *= 2.0
            if upper_bound > 1.0e4:
                raise ValueError(
                    "Could not match the requested kernel_standard_deviation."
                )

        return brentq(objective, 0.0, upper_bound)

    def _compute_kernel_standard_deviation_from_coefficient(self, kernel_coefficient):
        """Return the kernel standard deviation for a specific parameter value."""
        if self.kernel_type == "top_hat":
            return kernel_coefficient / np.sqrt(3.0)

        lower_bound = -0.5 * self.length
        upper_bound = 0.5 * self.length

        first_moment = self._integrate_f(
            lambda x: x * float(
                self._evaluate_kernel_with_coefficient(x, kernel_coefficient)
            ),
            lower_bound,
            upper_bound,
        )
        second_moment = self._integrate_f(
            lambda x: (x**2) * float(
                self._evaluate_kernel_with_coefficient(x, kernel_coefficient)
            ),
            lower_bound,
            upper_bound,
        )

        variance = max(second_moment - first_moment**2, 0.0)
        return np.sqrt(variance)

    def _validate_kernel_inputs(self):
        """Validate the chosen kernel type and its parameter."""
        if not np.isfinite(self.kernel_coefficient):
            raise ValueError("kernel_coefficient must be finite.")

        if self.kernel_type == "top_hat":
            if self.kernel_coefficient <= 0.0:
                raise ValueError("Top-hat width must be positive.")
            if self.kernel_coefficient > 0.5 * self.length:
                raise ValueError("Top-hat width must be at most half the domain length.")
            return

        if self.kernel_type == "von_mises":
            if self.kernel_coefficient < 0.0:
                raise ValueError("Von Mises concentration must be non-negative.")
            return

        raise ValueError("Unsupported kernel type.")

    def _compute_dx(self):
        """Return the spatial grid spacing."""
        return self.length / self.number_of_points

    def _build_space_grid(self):
        """Build the periodic spatial grid used by the FFT solver."""
        return np.linspace(
            self.a_border,
            self.b_border,
            self.number_of_points,
            endpoint=False,
        )

    def _build_time_grid(self):
        """Build the array of stored output times."""
        return np.arange(self.number_of_steps + 1, dtype=float) * self.dt

    def _build_wavenumbers(self):
        """Build the Fourier wavenumbers associated with the spatial grid."""
        return 2.0 * np.pi * np.fft.fftfreq(self.number_of_points, d=self.dx)

    def _compute_rk4_substeps(self):
        """Choose enough RK4 substeps to keep the explicit diffusion update stable."""
        max_diffusion = float(np.max(self.coefficient_diffusion))
        max_wavenumber_squared = float(np.max(self.wavenumbers**2))
        stability_scale = self.dt * max_diffusion * max_wavenumber_squared
        stable_limit = 2.5

        if stability_scale <= stable_limit:
            return 1

        return int(np.ceil(stability_scale / stable_limit))

    def _initialise_solution_storage(self):
        """Allocate the array that stores the full time history of the solution."""
        return np.zeros(
            (self.number_of_steps + 1, self.number_of_points, self.number_of_population),
            dtype=float,
        )

    def _evaluate_initial_condition(self):
        """Evaluate the user profile or build a default evenly spaced Gaussian one."""
        if self.initial_condition is None:
            return self._build_default_initial_condition()

        values = np.asarray(self.initial_condition(self.x), dtype=float)

        if values.shape == (self.number_of_points,) and self.number_of_population == 1:
            return values[:, np.newaxis]

        if values.shape == (self.number_of_population, self.number_of_points):
            return values.T

        if values.shape != (self.number_of_points, self.number_of_population):
            raise ValueError(
                "initial_condition(x) must return an array of shape "
                "(number_of_points, number_of_population) or "
                "(number_of_population, number_of_points)."
            )

        return values

    def _build_default_initial_condition(self):
        """Build one Gaussian bump per population, evenly spaced on the domain."""
        centers = self._build_default_population_centers()
        gaussian_width = self._compute_default_gaussian_width()
        profiles = [
            self._build_periodic_gaussian(center, gaussian_width)
            for center in centers
        ]
        values = np.column_stack(profiles)
        return self._normalise_population_profiles(values)

    def _build_default_population_centers(self):
        """Return evenly spaced Gaussian centers across the periodic interval."""
        spacing = self.length / self.number_of_population
        return self.a_border + (0.5 + np.arange(self.number_of_population)) * spacing

    def _compute_default_gaussian_width(self):
        """Choose a Gaussian width small enough to separate neighbouring populations."""
        spacing = self.length / self.number_of_population
        return max(spacing / 6.0, self.dx)

    def _build_periodic_gaussian(self, center, width):
        """Evaluate a Gaussian profile using periodic distance to its center."""
        wrapped_distance = (
            (self.x - center + 0.5 * self.length) % self.length
        ) - 0.5 * self.length
        return np.exp(-0.5 * (wrapped_distance / width) ** 2)

    def _normalise_population_profiles(self, values):
        """Scale each population profile so all default masses are equal to one."""
        masses = self.dx * np.sum(values, axis=0, keepdims=True)
        return values / masses

    def _build_kernel_grid(self):
        """Build offsets centered on zero for sampling the periodic kernel."""
        offsets = np.arange(self.number_of_points, dtype=float) * self.dx
        return np.where(offsets < 0.5 * self.length, offsets, offsets - self.length)

    def _generate_kernel_values(self):
        """Sample the chosen kernel on the FFT grid and normalise its mass."""
        kernel_grid = self._build_kernel_grid()

        kernel_values = self._evaluate_kernel(kernel_grid)

        return self._normalise_kernel_values(kernel_values)

    def _top_hat_kernel(self, values, kernel_coefficient):
        """Evaluate the top-hat kernel on the given offsets."""
        values = np.asarray(values, dtype=float)
        return np.where(
            np.abs(values) <= kernel_coefficient,
            1.0 / (2.0 * kernel_coefficient),
            0.0,
        )

    def _von_mises_kernel(self, values, kernel_coefficient):
        """Evaluate the periodic von Mises kernel on the given offsets."""
        values = np.asarray(values, dtype=float)
        phase = 2.0 * np.pi * values / self.length
        denominator = self.length * i0e(kernel_coefficient)
        exponent = kernel_coefficient * (np.cos(phase) - 1.0)
        return np.exp(exponent) / denominator

    def _evaluate_kernel_with_coefficient(self, values, kernel_coefficient):
        """Evaluate the selected kernel using an explicit parameter value."""
        if self.kernel_type == "top_hat":
            return self._top_hat_kernel(values, kernel_coefficient)
        return self._von_mises_kernel(values, kernel_coefficient)

    def _evaluate_kernel(self, values):
        """Evaluate the selected kernel on scalar or array inputs."""
        return self._evaluate_kernel_with_coefficient(values, self.kernel_coefficient)

    def _normalise_kernel_values(self, kernel_values):
        """Rescale the sampled kernel so its discrete integral is one."""
        normalisation = np.sum(kernel_values) * self.dx
        if normalisation <= 0.0:
            raise ValueError("Kernel must have positive mass.")
        return kernel_values / normalisation

    def _fft_vector(self, values):
        """Compute the FFT of a one-dimensional array."""
        return np.fft.fft(values)

    def _ifft_vector(self, values):
        """Compute the inverse FFT of a one-dimensional array and keep the real part."""
        return np.fft.ifft(values).real

    def _fft_matrix(self, values):
        """Compute the FFT along the spatial axis for all populations."""
        return np.fft.fft(values, axis=0)

    def _ifft_matrix(self, values):
        """Compute the inverse FFT along the spatial axis for all populations."""
        return np.fft.ifft(values, axis=0).real

    def _first_derivative_multiplier(self):
        """Return the Fourier multiplier for a first spatial derivative."""
        return 1j * self.wavenumbers

    def _second_derivative_multiplier(self):
        """Return the Fourier multiplier for a second spatial derivative."""
        return -(self.wavenumbers ** 2)

    def _apply_first_derivative_fourier(self, values_hat):
        """Differentiate Fourier coefficients once in space."""
        multiplier = self._first_derivative_multiplier()[:, np.newaxis]
        return multiplier * values_hat

    def _apply_second_derivative_fourier(self, values_hat):
        """Differentiate Fourier coefficients twice in space."""
        multiplier = self._second_derivative_multiplier()[:, np.newaxis]
        return multiplier * values_hat

    def _compute_fourier_coefficients(self, values):
        """Convert population values from physical space to Fourier space."""
        return self._fft_matrix(values)

    def _compute_physical_values(self, values_hat):
        """Convert Fourier coefficients back to physical space."""
        return self._ifft_matrix(values_hat)

    def _convolve_in_fourier_space(self, values_hat):
        """Apply the kernel convolution using the convolution theorem."""
        kernel_hat = self.kernel_fourier[:, np.newaxis]
        return self.dx * kernel_hat * values_hat

    def _compute_smoothed_density(self, population):
        """Compute the non-local averaged densities for each population."""
        population_hat = self._compute_fourier_coefficients(population)
        smoothed_hat = self._convolve_in_fourier_space(population_hat)
        return self._compute_physical_values(smoothed_hat)

    def _compute_smoothed_density_gradient(self, population):
        """Compute the spatial gradient of the non-local averaged densities."""
        population_hat = self._compute_fourier_coefficients(population)
        smoothed_hat = self._convolve_in_fourier_space(population_hat)
        gradient_hat = self._apply_first_derivative_fourier(smoothed_hat)
        return self._compute_physical_values(gradient_hat)

    def _apply_interaction_matrix(self, values):
        """Mix species contributions through the interaction matrix."""
        return values @ self.coefficient_attraction.T

    def _compute_advective_velocity(self, population):
        """Build the non-local advective velocity field."""
        smoothed_gradient = self._compute_smoothed_density_gradient(population)
        return self._apply_interaction_matrix(smoothed_gradient)

    def _compute_flux(self, population):
        """Compute the advective flux for each population."""
        velocity = self._compute_advective_velocity(population)
        return population * velocity

    def _compute_flux_derivative(self, population):
        """Compute the spatial derivative of the advective flux."""
        flux = self._compute_flux(population)
        flux_hat = self._compute_fourier_coefficients(flux)
        derivative_hat = self._apply_first_derivative_fourier(flux_hat)
        return self._compute_physical_values(derivative_hat)

    def _compute_diffusion_term(self, population):
        """Compute the diffusion contribution to the PDE right-hand side."""
        population_hat = self._compute_fourier_coefficients(population)
        second_derivative_hat = self._apply_second_derivative_fourier(population_hat)
        second_derivative = self._compute_physical_values(second_derivative_hat)
        return second_derivative * self.coefficient_diffusion[np.newaxis, :]

    def _compute_rhs(self, population):
        """Assemble the full PDE right-hand side in physical space."""
        diffusion_term = self._compute_diffusion_term(population)
        advection_term = self._compute_flux_derivative(population)
        return diffusion_term - advection_term

    def _runge_kutta_increment(self, population, slope, factor, dt_step):
        """Build an intermediate RK4 state from a slope estimate."""
        return population + factor * dt_step * slope

    def _runge_kutta_step(self, population, dt_step):
        """Advance the solution by one time step using classical RK4."""
        k1 = self._compute_rhs(population)
        k2 = self._compute_rhs(
            self._runge_kutta_increment(population, k1, 0.5, dt_step)
        )
        k3 = self._compute_rhs(
            self._runge_kutta_increment(population, k2, 0.5, dt_step)
        )
        k4 = self._compute_rhs(
            self._runge_kutta_increment(population, k3, 1.0, dt_step)
        )
        return population + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _advance_one_output_step(self, population):
        """Advance one stored time step, using internal RK4 substeps when needed."""
        next_population = population
        for _ in range(self.rk4_substeps):
            next_population = self._runge_kutta_step(next_population, self.internal_dt)
        return next_population

    def step(self, population):
        """Advance a provided state by one solver time step."""
        return self._advance_one_output_step(population)

    def solve(self):
        """Run the full simulation and store each time step."""
        for step_index in range(self.number_of_steps):
            next_state = self._advance_one_output_step(self.U[step_index, :, :])
            self.U[step_index + 1, :, :] = next_state
            self.U_fourier[step_index + 1, :, :] = self._compute_fourier_coefficients(
                next_state
            )

        self._solution_computed = True
        return self.time, self.U

    def get_solution(self):
        """Return the stored solution in physical space."""
        return self.U

    def get_fourier_solution(self):
        """Return the stored solution in Fourier space."""
        return self.U_fourier

    def get_snapshot(self, time_index):
        """Return the solution at one stored time index."""
        return self.U[time_index, :, :]

    def get_final_distribution(self):
        """Return the final stored state in physical space, solving if needed."""
        self._ensure_solution_computed()
        return self.U[-1, :, :].copy()

    def get_final_initial_condition(self):
        """Return an initial-condition callback built from the final stored state."""
        final_distribution = self.get_final_distribution()
        return self._build_initial_condition_from_state(final_distribution)

    def get_mass(self):
        """Return the total mass of each population at each stored time."""
        return self.dx * np.sum(self.U, axis=1)
    
    def get_kernel_standard_deviation(self):
        """Return the standard deviation of the kernel as a measure of its width."""
        return self._compute_kernel_standard_deviation_from_coefficient(
            self.kernel_coefficient
        )

    def _integrate_f(self, function, lower_bound, upper_bound):
        """Helper method to integrate a function over the periodic domain."""
        return quad(function, lower_bound, upper_bound)[0]

    def _ensure_solution_computed(self):
        """Run the solver once before using any visualization based on the full solution."""
        if not self._solution_computed:
            self.solve()

    def _resolve_output_path(self, save_path, default_filename):
        """Return a writable output path and create missing parent directories."""
        output_path = Path(save_path) if save_path is not None else Path(default_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _compute_figure_layout(self, number_of_plots):
        """Choose a near-square subplot layout for the requested number of plots."""
        columns = int(np.ceil(np.sqrt(number_of_plots)))
        rows = int(np.ceil(number_of_plots / columns))
        return rows, columns

    def _select_snapshot_indices(self, number_of_plots):
        """Pick evenly spaced stored times, including the first and last snapshots."""
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
        """Build a common y-range for all plots so amplitudes are comparable."""
        solution_min = float(np.min(self.U))
        solution_max = float(np.max(self.U))
        amplitude = solution_max - solution_min
        margin = 0.05 * amplitude if amplitude > 0.0 else 0.1 * max(solution_max, 1.0)
        return solution_min - margin, solution_max + margin

    def _build_initial_condition_from_state(self, state):
        """Create a periodic initial-condition callback from one stored state."""
        state = np.asarray(state, dtype=float)
        expected_shape = (self.number_of_points, self.number_of_population)
        if state.shape != expected_shape:
            raise ValueError(
                "state must have shape "
                "(number_of_points, number_of_population)."
            )

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
            return np.column_stack(interpolated_profiles)

        return initial_condition

    def _plot_population_profiles(self, axis, state, title):
        """Draw one spatial profile per population on a given matplotlib axis."""
        for population_index in range(self.number_of_population):
            axis.plot(
                self.x,
                state[:, population_index],
                label=f"Population {population_index + 1}",
            )

        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("density")

    def create_solution_gif(
        self,
        interval=100,
        save=False,
        save_path=None,
        fps=15,
    ):
        """Create an animated GIF of the solution and optionally save it."""
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

    def plot_solution_snapshots(
        self,
        number_of_plots=4,
        save=False,
        save_path=None,
    ):
        """Create a figure with evenly spaced solution snapshots and optionally save it."""

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

    def plot_solution_heatmaps(
        self,
        cmap="hot_r",
        share_color_scale=True,
        save=False,
        save_path=None,
    ):
        """Create one x-t density heatmap per population and optionally save it."""
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
            axis.set_title(f"$u_{population_index + 1}$")
            axis.set_xlabel("x")
            if population_index == 0:
                axis.set_ylabel("t")

            colorbar = figure.colorbar(image, ax=axis)
            colorbar.set_label("density")

        figure.tight_layout()

        if save:
            output_path = self._resolve_output_path(save_path, "solution_heatmaps.png")
            figure.savefig(output_path, bbox_inches="tight")

        return figure, axes
    
    def get_kernel_info(self):
        """Return a dictionary of kernel properties for reference."""
        return {
            "type": self.kernel_type,
            "coefficient": self.kernel_coefficient,
            "standard_deviation": self.get_kernel_standard_deviation(),
        }