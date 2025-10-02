import sys
sys.path.append('code')
import copy
import numpy as np
import lst as lst
import example_ae_setting as dyn_setting
import scipy
from functools import partial
import matplotlib.pyplot as plt



def simulate(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, t_step, pert):

    x_init = [mbar, k_3]

    ndof = 4
    # get initial condition from LST
    f_int_dict = {"obj": dyn_setting.obj}
    f_pint_pw_dict = {"obj": dyn_setting.pobj_pw}
    f_pint_px_dict = {"obj": dyn_setting.pobj_px}

    f_int_top_dict = {"LST": dyn_setting.f_LST}
    f_pint_top_pv_dict = {"LST": dyn_setting.f_pLST_pv}

    lst_obj = lst.lst(
        ndof,
        x_init,
        dyn_setting.res,
        f_int_dict,
        dyn_setting.pres_pw,
        dyn_setting.pres_px,
        f_pint_pw_dict,
        f_pint_px_dict,
        None,
        None,
        f_int_top_dict,
        f_pint_top_pv_dict,
    )
    lst_obj.solve()

    v_init = lst_obj.get_v()
    print("real:", v_init[-2], "imag", v_init[-1])

    w_init_0 = lst_obj.get_w()


    from example_ae_setting_2 import ae_set

    # create dyn setting instance
    dyn_setting_2 = ae_set(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con)
    class DynSetting:
        def __init__(self, theta, x):
            self.theta = theta
            self.x = x

        def res(self, t, w):
            x = self.x
            theta = self.theta
            return dyn_setting_2.res(w, x, theta)



    sys_inst = DynSetting(0, x_init)

    # perturb state variable
    w_init_init = copy.deepcopy(w_init_0)
    w_init_init[1] += 1e-3

    # set integration bounds
    t0 = 0.0
    t_bound = 1000.0
    # t_step = 0.005
    # t_step = 0.1
    N = int(t_bound / t_step) 


    # create integrator instance
    integrator_init = scipy.integrate.RK45(sys_inst.res, t0, w_init_init, t_bound, max_step=t_step, rtol=0.001, atol=1e-06, vectorized=False, first_step=None)

    if pert:
        # create perturbation in theta
        pert_mag = 7
        th_pert = np.linspace(0, (pert_mag) * np.pi/180.0, 5)
        th_per_2 = np.linspace((pert_mag) * np.pi/180.0, 0, 5)
        th_pert = np.concatenate((th_pert, th_per_2))

        t_start_pert = int((500) / t_step)
        t_end_pert = t_start_pert + 10

    # collect data
    t_values_init = []
    y1_init = []
    y2_init = []
    y3_init = []
    y4_init = []
    for i in range(N):
        # if time steps are within perturbation window, apply perturbation
        if pert:
            if i >= t_start_pert and i < t_end_pert:
                sys_inst.theta = th_pert[i - t_start_pert]

        integrator_init.step()

        t_values_init.append(integrator_init.t)
        y1_init.append(integrator_init.y[0])
        y2_init.append(integrator_init.y[1])
        y3_init.append(integrator_init.y[2])
        y4_init.append(integrator_init.y[3])
        if integrator_init.status == 'finished':
            break

    # save data: time, h, alpha, dh/dt, dalpha/dt
    tostack = [np.array(t_values_init), np.array(y1_init), np.array(y2_init), np.array(y3_init), np.array(y4_init)]
    timeseries_data_smalltheta = np.stack((tostack), axis=1)
    # np.savetxt('timeseries_data_theta_first.csv', timeseries_data_smalltheta)
    return timeseries_data_smalltheta

    # import niceplots
    # custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263', '#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
    # c1 = custom_colors[0]
    # c2 = custom_colors[1]
    # c3 = custom_colors[2]
    # c4 = custom_colors[3]
    # c5 = custom_colors[4]
    # c6 = custom_colors[5]

    # plt.rcParams["axes.spines.top"] = False
    # plt.rcParams["axes.spines.right"] = False

    # labelpad = 45
    # fontsize = 16

    # plt.rcParams["font.size"] = fontsize
    # plt.rcParams["axes.titlesize"] = fontsize
    # plt.rcParams["axes.labelsize"] = fontsize
    # plt.rcParams["xtick.labelsize"] = fontsize
    # plt.rcParams["ytick.labelsize"] = fontsize

    # plt.rcParams["mathtext.rm"] = "serif"
    # plt.rcParams["mathtext.it"] = "serif:italic"
    # plt.rcParams["mathtext.bf"] = "serif:bold"
    # plt.rcParams["mathtext.fontset"] = "custom"

    # fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
    # plt.style.use(niceplots.get_style('james-light'))

    # start_ind = int(np.round((t_start_pert) * t_step - 100))
    # end_ind = int(np.round((t_end_pert) * t_step + 100))
    # start_ind_t = int(start_ind/t_step)
    # end_ind_t = int(end_ind/t_step)

    # ax[0].plot(t_values_init, y1_init, '-', label=r"$\bar{h}$", color=c1)
    # ax[0].set_xlabel(r"$t$", fontsize=20)
    # ax[0].set_ylabel(r"$\bar{h}$", fontsize=18, rotation=0)
    # # ax[0].set_xlim([start_ind,end_ind])
    # ax[0].xaxis.set_visible(False)
    # ax[0].yaxis.set_label_coords(-0.1,0.5)

    # ax[1].plot(t_values_init, y2_init, '-', label=r"$\alpha$", color=c2)
    # ax[1].set_xlabel(r"$t$", fontsize=20)
    # ax[1].set_ylabel(r"$\alpha$", fontsize=18, rotation=0)
    # # ax[1].set_xlim([start_ind,end_ind])
    # ax[1].yaxis.set_label_coords(-0.1,0.5)

    # plt.savefig("unsteady_ae_pert_theta_first.pdf")

# plt.show()

