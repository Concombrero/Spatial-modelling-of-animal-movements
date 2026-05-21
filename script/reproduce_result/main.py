import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from solver import Model1D


def main():
    
    n = 2 # number of population
    # attraction matrix between populations
    coefficient_attraction = np.array([[0, 1], 
                                       [-1, 0]]) 
    coefficient_diffusion = np.array([0.01, 0.01]) # diffusion coefficient for each population
    solver = Model1D(
        a_border = 0.0,
        b_border = 1.0,
        number_of_points = 128,
        total_time = 0.5,
        dt = 1e-4,
        number_of_population = n,
        coefficient_attraction = coefficient_attraction,
        coefficient_diffusion = coefficient_diffusion,
        kernel_type = "von_mises",
        kernel_standard_deviation=0.05
        
    )

    figure, anim = solver.create_solution_gif(interval = 10)
    plt.show()
    
    

if __name__ == "__main__":
    main()