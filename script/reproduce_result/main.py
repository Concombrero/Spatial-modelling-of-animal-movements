import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from solver import Model1D

def gaussian_initial_condition(x):
    """Build two periodic Gaussian populations centered at 0 and 1/2."""
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    centers = np.array([0.0, 0.5 * length])
    gaussian_width = max(length / 12.0, dx)

    profiles = []
    for center in centers:
        wrapped_distance = ((x - center + 0.5 * length) % length) - 0.5 * length
        profiles.append(np.exp(-0.5 * (wrapped_distance / gaussian_width) ** 2))

    values = np.column_stack(profiles)
    masses = dx * np.sum(values, axis=0, keepdims=True)
    return values / masses

def paper_initial_condition(
    x,
    seed=0,
    perturbation_amplitude=0.05,
    smoothing_length=0.08,
):
    """Build a smooth random perturbation of the homogeneous steady state."""
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
    length = (x[-1] - x[0]) + dx
    number_of_points = x.size

    rng = np.random.default_rng(seed)
    base_state = np.ones((number_of_points, 2), dtype=float)
    raw_noise = rng.standard_normal((number_of_points, 2))

    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(number_of_points, d=dx)
    smoothing_filter = np.exp(-0.5 * (smoothing_length * wavenumbers) ** 2)
    smooth_noise = np.fft.ifft(
        np.fft.fft(raw_noise, axis=0) * smoothing_filter[:, np.newaxis],
        axis=0,
    ).real

    smooth_noise -= np.mean(smooth_noise, axis=0, keepdims=True)
    noise_scale = np.max(np.abs(smooth_noise), axis=0, keepdims=True)
    smooth_noise /= np.where(noise_scale > 0.0, noise_scale, 1.0)

    values = base_state + perturbation_amplitude * smooth_noise
    values = np.clip(values, 1.0e-12, None)

    masses = dx * np.sum(values, axis=0, keepdims=True)
    return values / masses

def fig1():
    n = 2 # number of population
    # attraction matrix between populations
    coefficient_attraction = np.array([[0, -2], 
                                       [-2, 0]]) 
    coefficient_diffusion = np.array([1, 1]) # diffusion coefficient for each population
    initial_condition = paper_initial_condition
    
    for sigma in [0.1, 0.05, 0.025]:
        simulation = Model1D(
            a_border = 0.0,
            b_border = 1.0,
            number_of_points= 128,
            total_time= 0.5,
            dt = 1e-4,
            initial_condition=initial_condition,
            number_of_population=n,
            coefficient_attraction=coefficient_attraction,
            coefficient_diffusion=coefficient_diffusion,
            kernel_type="von_mises",
            kernel_standard_deviation=sigma
        )
        
        figure, axes = simulation.plot_solution_heatmaps(save = True, save_path=f"./script/reproduce_result/fig1/heatmap_sigma_{sigma}.png")
        
        figure, annimation = simulation.create_solution_gif(interval=1, fps= 60, save = True, save_path=f"./script/reproduce_result/fig1/animation_sigma_{sigma}.gif")
        initial_condition = simulation.get_final_initial_condition()

def fig2_sigma(sigma, time, repetition, initial_condition = paper_initial_condition):
    n = 2 # number of population
    # attraction matrix between populations
    coefficient_attraction = np.array([[1.5, -1.5], 
                                       [1.5, 1.5]]) 
    coefficient_diffusion = np.array([1, 1]) # diffusion coefficient for each population
    for i in range(repetition):
            simulation = Model1D(
                a_border = 0.0,
                b_border = 1.0,
                number_of_points= 128,
                total_time= time,
                dt = 1e-4,
                initial_condition=initial_condition,
                number_of_population=n,
                coefficient_attraction=coefficient_attraction,
                coefficient_diffusion=coefficient_diffusion,
                kernel_type="von_mises",
                kernel_standard_deviation=sigma
            )
            
            figure, axes = simulation.plot_solution_heatmaps(save = True, save_path=f"./script/reproduce_result/fig2/heatmap_sigma_{sigma}_repetition_{i}.png")
            plt.show()
            initial_condition = simulation.get_final_initial_condition()
    
    return initial_condition
    

def fig2():
    new_initial_condition = fig2_sigma(sigma=0.1, time=0.3, repetition=3)
    new_initial_condition = fig2_sigma(sigma=0.05, time=0.3, repetition=3, initial_condition=new_initial_condition)
    
def main():
    fig2()
    
if __name__ == "__main__":
    main()
