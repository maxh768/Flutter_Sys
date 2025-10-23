import numpy as np
# from scipy import optimize
import copy
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from matplotlib import colors
from example_setting_ae_forbif import ae_forbif
import Hopf_bif
from plot_packages import *
from LCO import LCO_TS, LCO_TS_stencil
import cmath

def generate_LCO(x0, ntimeinstance, ndof, N, magmin, magmax, pha_0, force, pforcepx, pforcepw, pforcepmu):

    ind_p = 1

    # Obtain Hopf bifurcation results
    def func(w, mu, x):

        return force(mu, w, x)
    
    def func_A(w, mu, x):

        return pforcepw(mu, w, x)

    Hopf_bif_obj = Hopf_bif.Hopf_bif(x0, func, func_A, ndof)
    mu_lower = 0.1
    mu_upper = 1.4
    delta = 1e-8
    Hopf_bif_obj.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

    # Hopf_bif_obj.mu = mu_con
    # Hopf_bif_obj.w = [0, 1e-3, 0, 0]
    # eigval, eigvec = Hopf_bif_obj.solve_eig_R(mu_con, Hopf_bif_obj.w)
    # eigval = np.imag(eigval) * 1j
    # Hopf_bif_obj.eigval_bif_R = eigval
    # Hopf_bif_obj.eigvec_bif_R = eigvec

    mu_hopf = Hopf_bif_obj.mu
    eigval_bif_hopf = Hopf_bif_obj.eigval_bif_R
    eigvec_bif_hopf = Hopf_bif_obj.eigvec_bif_R

    # Convert to time domain for initialization
    w0 = np.zeros(ntimeinstance * ndof)

    eigmag = np.zeros(ndof)
    eigpha = np.zeros(ndof)

    for i in range(ndof):
        eigmag[i] = abs(eigvec_bif_hopf[i])
        eigpha[i] = cmath.phase(eigvec_bif_hopf[i])

    eigpha0 = eigpha[ind_p]
    for i in range(ndof):

        eigpha[i] -= eigpha0

    ratio = magmin / eigmag[ind_p]

    print("Hopf_bif_obj.w", Hopf_bif_obj.w)
    print("eigvec_bif_hopf", eigvec_bif_hopf)
    for i in range(ntimeinstance):
        for j in range(ndof):

            mag_loc = eigmag[j] * ratio
            pha_loc = eigpha[j] + pha_0 + (float(i) / float(N)) * (2.0 * np.pi)

            w0[i * ndof + j] = mag_loc * np.sin(pha_loc) + Hopf_bif_obj.w[j]
            
    print("mu_hopf", mu_hopf)
    print("eigval_bif_hopf", eigval_bif_hopf)
    print("w0", w0)
    
    oscillator = LCO_TS(
        force, ntimeinstance, ndof, x=x0, pforcepx_func=pforcepx, pforcepw_func=pforcepw, pforcepmu_func=pforcepmu
    )
    oscillator.set_motion_mag_pha(magmin, pha_0, ind_p=ind_p)


    xin = np.zeros(ntimeinstance * ndof + 2)
    xin[0] = mu_hopf
    xin[1] = np.imag(eigval_bif_hopf)
    xin[2:] = w0[:]

    w0 = oscillator.solve(xin)
    # np.savetxt("w0", w0)

    magmin = magmin
    magmax = magmax

    mag_arr, mu_arr = oscillator.gen_bif_diag(xin, magmin, magmax, N)

    return mag_arr, mu_arr


kappa_5_con = 50.0
Omega_con = 0.5
xa_con = 0.2
ra_con = 0.3
mu_con = 0.8
a_con = -0.3
theta_con = 0
mbar = 12
k_3 = -1

dyn_setting = ae_forbif(kappa_5_con, Omega_con, ra_con, xa_con, a_con)
x = [mbar, k_3]

k3_vals = np.linspace(-3, 1, 20)

x_path = np.zeros((len(k3_vals), 2))
x_path[:, 0] = np.ones_like(k3_vals) * mbar
x_path[:, 1] = k3_vals

x_path = x_path[::5, :]
x_path = np.concatenate((x_path, np.array([[mbar, 1]])), axis=0)

ntimeinstance = 7
ndof = 4

N = 50
magmin = 0.03
magmax = 0.3
pha_0 = 0.2

NN = x_path.shape[0]
mag_arr_list = []
mu_arr_list = []
k3_arr = []
for i in range(NN):

    print("i", i)

    x_init_inter = x_path[i, :]
    k3_arr.append(x_init_inter[1])

    mag_arr, mu_arr = generate_LCO(
        x_init_inter,
        ntimeinstance,
        ndof,
        N,
        magmin,
        magmax,
        pha_0,
        dyn_setting.force,
        None,
        dyn_setting.pforcepw,
        None,
    )

    mag_arr_list.append(mag_arr)
    mu_arr_list.append(mu_arr)


fig, axs = plt.subplots(1, figsize=(8, 4))


def plot(ax, mu_arr, mag_arr, color):

    # Bifurcation diagram
    axs.plot(mu_arr, mag_arr, "-", color=color)

    return ax


rgb_blue = colors.to_rgba(my_blue)
rgb_green = colors.to_rgba(my_green)
for i in range(NN):

    weight = 1.0 - float(i) / float(NN)

    my_color_arr = np.zeros(3)
    for j in range(3):
        my_color_arr[j] = rgb_blue[j] * weight + rgb_green[j] * (1 - weight)

    my_color = (my_color_arr[0], my_color_arr[1], my_color_arr[2], 1.0)

    mag_arr = mag_arr_list[i]
    mu_arr = mu_arr_list[i]

    axs = plot(axs, mu_arr, mag_arr, my_color)
    # HACK
    # axs = plot(axs, mu_arr - mu_arr[0], mag_arr, my_color)


    axs.text(mu_arr[-1], mag_arr[-1], f'k3 = {k3_arr[i]:.2f}', color=my_color, fontsize=12)
    # HACK
    # axs.text(mu_arr[-1] - mu_arr[0], mag_arr[-1], str(i), color=my_color, fontsize=20)

axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)

axs.set_xlabel(r"$\mu$", fontsize=20)
axs.set_ylabel(r"$|{w}_{2,1}|$", fontsize=20, rotation=0)
axs.yaxis.set_label_coords(-0.175, 0.4)

mu_arr_init = mu_arr_list[0]
mu_arr_final = mu_arr_list[-1]
mag_arr_init = mag_arr_list[0]
mag_arr_final = mag_arr_list[-1]


min_mu = min(min(mu_arr_init), min(mu_arr_final))
max_mu = max(max(mu_arr_init), max(mu_arr_final))
delta_mu = max_mu - min_mu
# axs.set_xlim([min_mu - delta_mu * 0.1, max_mu + delta_mu * 0.1])
axs.set_ylim([0, magmax * 1.1])

# Additional lines
index = 9
# axs.plot([min_mu - delta_mu * 0.1, mu_arr_final[index]], [0.5, 0.5], color="k", alpha=0.2)
# axs.plot([mu_arr_init[index], mu_arr_init[index]], [mag_arr_init[index], 0], color=my_blue, alpha=0.2)
# axs.plot([mu_arr_final[index], mu_arr_final[index]], [mag_arr_final[index], 0], color=my_green, alpha=0.2)

# axs.text(1.02, 0.34, r"Baseline", fontsize=20, color=my_blue)
# axs.text(1.15, 0.34, r"Optimized", fontsize=20, color=my_green)

plt.savefig("example_bif_diagram.pdf", bbox_inches="tight")


