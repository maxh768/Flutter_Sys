import numpy as np
from scipy.integrate import solve_ivp
import example_ae_setting as dyn_setting
import matplotlib.pyplot as plt

x_init = [12.0, -2.5] # design

# Define your nonlinear system f(x)
def f(w):
    return dyn_setting.res(w, x_init)

# Define the Jacobian matrix Df(x)
def Df(w):
    Ab = dyn_setting.evalA(x_init)
    pFpw = dyn_setting.evalpFnlpw(w, x_init)
    return Ab + pFpw


# Function to define the augmented ODE system for integration
def augmented_ode(t, Y_aug, f_func, Df_func, n):
    # Y_aug is a 1D array of size n + n*n
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
T = 5.3097 # Period

# Initial condition for the periodic orbit
x0_orbit = np.array([0.04664931, -0.02250585, -0.05325736, 0.27486855])

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
    method='RK45', # A good general-purpose solver
    rtol=1e-8, # Relative tolerance
    atol=1e-10 # Absolute tolerance
)

# Extract the final state at t=T
Y_at_T = sol.y[:, -1] # Last column contains the state at T
print(sol.y.shape)
print(sol.t.shape)

# Extract the periodic orbit state at T (should be close to x0_orbit)
x_at_T = Y_at_T[0:n]

# Extract the fundamental matrix at T (Monodromy Matrix)
monodromy_matrix_flat = Y_at_T[n : n + n*n]
Monodromy_Matrix = monodromy_matrix_flat.reshape((n, n))

print("Monodromy Matrix M:")
print(Monodromy_Matrix)

# Calculate Floquet Multipliers
floquet_multipliers = np.linalg.eigvals(Monodromy_Matrix)

print("\nFloquet Multipliers:")
print(floquet_multipliers)

print("\nMagnitudes of Floquet Multipliers:")
print(np.abs(floquet_multipliers))

import niceplots
custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263', '#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
c1 = custom_colors[0]
c2 = custom_colors[1]
c3 = custom_colors[2]
c4 = custom_colors[3]
c5 = custom_colors[4]
c6 = custom_colors[5]

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

labelpad = 45
fontsize = 16

plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize

plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.fontset"] = "custom"

fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
plt.style.use(niceplots.get_style('james-light'))

ax[0].plot(sol.t, sol.y[0,:], '-', label=r"$\bar{h}$", color=c1)
ax[0].set_xlabel(r"$t$", fontsize=20)
ax[0].set_ylabel(r"$\bar{h}$", fontsize=18, rotation=0)
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_label_coords(-0.1,0.5)

ax[1].plot(sol.t, sol.y[1,:], '-', label=r"$\alpha$", color=c2)
ax[1].set_xlabel(r"$t$", fontsize=20)
ax[1].set_ylabel(r"$\alpha$", fontsize=18, rotation=0)
ax[1].yaxis.set_label_coords(-0.1,0.5)

plt.savefig("pert_test.pdf")

