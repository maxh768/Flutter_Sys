import numpy as np
from scipy.integrate import solve_ivp
from example_ae_setting_2 import ae_set
import matplotlib.pyplot as plt

def get_floquet(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, T, x0_orbit):
    dyn_setting = ae_set(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con)

    x_init = [mbar, k_3]

    # residual
    def f(w):
        return dyn_setting.res(w, x_init, theta_con)
    # jacobian
    def Df(w):
        Ab = dyn_setting.evalA(x_init)
        pFpw = dyn_setting.evalpFnlpw(w, x_init, theta_con)
        return Ab + pFpw


    # Function to define the augmented ODE system for integration
    def augmented_ode(t, Y_aug, f_func, Df_func, n):
        x = Y_aug[0:n]
        
        # Calculate f(x)
        x_dot = f_func(x)
        
        # Calculate Df(x)
        jacobian = Df_func(x)
        
        # Calculate d(delta_x)/dt for each perturbation
        delta_x_dots = np.zeros(n * n)
        for i in range(n):
            delta_x_i = Y_aug[n + i*n : n + (i+1)*n]
            delta_x_dots[i*n : (i+1)*n] = jacobian @ delta_x_i
            
        return np.concatenate((x_dot, delta_x_dots))

    # --- Main calculation ---
    n = 4  # Dimension of the system
    # T = 5.3097 # Period

    # Initial condition for the periodic orbit
    # x0_orbit = np.array([0.04664931, -0.02250585, -0.05325736, 0.27486855])

    # Initial conditions for the perturbations (columns of identity matrix)
    delta_x0_perturbations = np.eye(n).flatten() 

    # Combine initial conditions
    Y0_aug = np.concatenate((x0_orbit, delta_x0_perturbations))

    # Time span for integration
    t_span = (0, T)

    # Integrate the augmented system
    # Using 'dense_output=True' if you want values at intermediate points
    sol = solve_ivp(
        fun=lambda t, Y: augmented_ode(t, Y, f, Df, n),
        t_span=t_span,
        y0=Y0_aug,
        method='RK45', 
        rtol=1e-8, 
        atol=1e-10 
    )

    # Extract the final state at t=T
    Y_at_T = sol.y[:, -1] # Last column contains the state at T
    # print(sol.y.shape)
    # print(sol.t.shape)

    # Extract the periodic orbit state at T (should be close to x0_orbit)
    x_at_T = Y_at_T[0:n]

    # Extract the fundamental matrix at T (Monodromy Matrix)
    monodromy_matrix_flat = Y_at_T[n : n + n*n]
    Monodromy_Matrix = monodromy_matrix_flat.reshape((n, n))

    # print("Monodromy Matrix M:")
    # print(Monodromy_Matrix)

    # Calculate Floquet Multipliers
    floquet_multipliers = np.linalg.eigvals(Monodromy_Matrix)

    return floquet_multipliers

