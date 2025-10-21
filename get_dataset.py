import numpy as np
from unsteady_pert_theta import simulate
from get_T import get_T
from floquet_2 import get_floquet
import Hopf_bif as hopf
from example_setting_ae_forbif import ae_forbif

# define parameters to sweep

# default params : 
kappa_5_con = 50.0
Omega_con = 0.5
xa_con = 0.2
ra_con = 0.3
mu_con = 0.8
a_con = -0.3
theta_con = 0 # 2 * np.pi/180.0
mbar = 12
k_3 = -1

# sweep mbar from 5 to 15
# sweep k_3 from -3 to 1
# sweep k_5 from 30 to 70
# sweep mu from 0.5 to 1.0
mbar_arr = np.linspace(5, 13, 20)
k_3_arr = np.linspace(-3, 1, 20)
k_5_arr = np.linspace(30, 70, 20)

dt_big = 0.1
dt_small = 0.005
# for each parameter set:
    # a. run unsteady simulation to get data (no perturbation) - large timstep
    # b. get the period from run_T
    # c. run floquet analysis to get multipliers
    # d. run unsteady simulation with perturbation - small timestep
    # e. get first Lyapunov coefficient and critical bifurcation parameter from Hopf_bif analysis
    # e . save data

main_folder = 'sweep'
mbar_folder = 'mbar_sweep'
k_3_folder = 'k3_sweep'
k_5_folder = 'k5_sweep'
ndof = 4

for mbar_i in mbar_arr:
    print(f'Running mbar = {mbar_i:.2f}')
    data = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar_i, k_3, dt_big, False)
    x0 = data[-1,1:]
    T = get_T(data)
    floquet_multipliers = get_floquet(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar_i, k_3, T, x0)
    data_pert = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar_i, k_3, dt_small, True)
    data_pert_2 = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar_i, k_3, dt_small, True, T_period=T, pert_2=True)
    data_pert_3 = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar_i, k_3, dt_small, True, T_period=T/2, pert_2=True)

    ae = ae_forbif(kappa_5_con, Omega_con, ra_con, xa_con, a_con)
    x = [mbar_i, k_3]
    
    Hopf_bif_obj = hopf.Hopf_bif(x, ae.func, ae.func_A, ndof)
    mu_lower = 0.3
    mu_upper = 1.1
    delta = 1e-6

    Hopf_bif_obj.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

    mu_crit = Hopf_bif_obj.mu

    Hopf_bif_obj.solve_eig_L()
    l = Hopf_bif_obj.compute_stab(ae.func_B, ae.func_C)

    mult_l_mu = np.concatenate((floquet_multipliers, [l, mu_crit], np.array([T])))

    np.savetxt(f'./{main_folder}/{mbar_folder}/data_mbar_{mbar_i:.2f}_pert_T4.csv', data_pert_2)
    np.savetxt(f'./{main_folder}/{mbar_folder}/data_mbar_{mbar_i:.2f}.csv', data_pert)
    np.savetxt(f'./{main_folder}/{mbar_folder}/data_mbar_{mbar_i:.2f}_pert_T8.csv', data_pert_3)
    np.savetxt(f'./{main_folder}/{mbar_folder}/flo_l_mu_mbar_{mbar_i:.2f}.csv', mult_l_mu)


for k_3_i in k_3_arr:
    print(f'Running k_3 = {k_3_i:.2f}')
    data = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3_i, dt_big, False)
    x0 = data[-1,1:]
    T = get_T(data)
    floquet_multipliers = get_floquet(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3_i, T, x0)
    data_pert = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3_i, dt_small, True)
    data_pert_2 = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3_i, dt_small, True, T_period=T, pert_2=True)
    data_pert_3 = simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3_i, dt_small, True, T_period=T/2, pert_2=True)

    ae = ae_forbif(kappa_5_con, Omega_con, ra_con, xa_con, a_con)
    x = [mbar, k_3_i]
    
    Hopf_bif_obj = hopf.Hopf_bif(x, ae.func, ae.func_A, ndof)
    mu_lower = 0.3
    mu_upper = 1.1
    delta = 1e-6

    Hopf_bif_obj.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

    mu_crit = Hopf_bif_obj.mu

    Hopf_bif_obj.solve_eig_L()
    l = Hopf_bif_obj.compute_stab(ae.func_B, ae.func_C)

    mult_l_mu = np.concatenate((floquet_multipliers, [l, mu_crit], np.array([T])))

    np.savetxt(f'./{main_folder}/{k_3_folder}/data_k3_{k_3_i:.2f}_pert_T4.csv', data_pert_2)
    np.savetxt(f'./{main_folder}/{k_3_folder}/data_k3_{k_3_i:.2f}.csv', data_pert)
    np.savetxt(f'./{main_folder}/{k_3_folder}/data_k3_{k_3_i:.2f}_pert_T8.csv', data_pert_3)
    np.savetxt(f'./{main_folder}/{k_3_folder}/flo_l_mu_k3_{k_3_i:.2f}.csv', mult_l_mu)


for k_5_i in k_5_arr:
    print(f'Running k_5 = {k_5_i:.2f}')
    data = simulate(k_5_i, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, dt_big, False)
    x0 = data[-1,1:]
    T = get_T(data)
    floquet_multipliers = get_floquet(k_5_i, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, T, x0)
    data_pert = simulate(k_5_i, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, dt_small, True)
    data_pert_2 = simulate(k_5_i, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, dt_small, True, T_period=T, pert_2=True)
    data_pert_3 = simulate(k_5_i, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, dt_small, True, T_period=T/2, pert_2=True)

    ae = ae_forbif(k_5_i, Omega_con, ra_con, xa_con, a_con)
    x = [mbar, k_3]
    
    Hopf_bif_obj = hopf.Hopf_bif(x, ae.func, ae.func_A, ndof)
    mu_lower = 0.3
    mu_upper = 1.1
    delta = 1e-6

    Hopf_bif_obj.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

    mu_crit = Hopf_bif_obj.mu

    Hopf_bif_obj.solve_eig_L()
    l = Hopf_bif_obj.compute_stab(ae.func_B, ae.func_C)

    mult_l_mu = np.concatenate((floquet_multipliers, [l, mu_crit], np.array([T])))

    np.savetxt(f'./{main_folder}/{k_5_folder}/data_k5_{k_5_i:.2f}_pert_T4.csv', data_pert_2)
    np.savetxt(f'./{main_folder}/{k_5_folder}/data_k5_{k_5_i:.2f}.csv', data_pert)
    np.savetxt(f'./{main_folder}/{k_5_folder}/data_k5_{k_5_i:.2f}_pert_T8.csv', data_pert_3)
    np.savetxt(f'./{main_folder}/{k_5_folder}/flo_l_mu_k5_{k_5_i:.2f}.csv', mult_l_mu)
