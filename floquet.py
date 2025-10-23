import numpy as np
from scipy.integrate import solve_ivp
from example_ae_setting_2 import ae_set
from example_setting_ae_forbif import ae_forbif
import matplotlib.pyplot as plt
import scipy

def get_floquet(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, T, x0_orbit):
    dyn_setting = ae_set(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con)

    x_init = [mbar, k_3]

    # residual
    def f(t, w):
        return dyn_setting.res(w, x_init, theta_con)
    
    # kappa_5, Omega, r_alpha, x_alpha, a)
    ae_bif = ae_forbif(kappa_5_con, Omega_con, ra_con, xa_con, a_con)
    
    # def linearize_system(w):
    #     """Compute the Jacobian matrix of the system at state w."""
    #     epsilon = 1e-6
    #     n = len(w)
    #     J = np.zeros((n, n))
    #     f0 = f(0, w)
    #     for i in range(n):
    #         w_perturbed = np.copy(w)
    #         w_perturbed[i] += epsilon
    #         f_perturbed = f(0, w_perturbed)
    #         J[:, i] = (f_perturbed - f0) / epsilon

    #     return J

    def linearize_system(w):
        A = ae_bif.func_A(w, mu_con, x_init)
        return A

    def euler_full(x0, T, dt):
        x1 = x0[0]
        x2 = x0[1]
        dx1 = x0[2]
        dx2 = x0[3]

        N = int(T / dt) + 1
        x_save = np.zeros((N+1, 4))
        x_save[0, :] = x0


        for i in range(N):
            x = np.array([x1, x2, dx1, dx2])
            d2x1, d2x2 = f(0, x)[2], f(0, x)[3]
            x1 += dx1 * dt
            x2 += dx2 * dt
            dx1 += d2x1 * dt
            dx2 += d2x2 * dt
            x_save[i+1, :] = np.array([x1, x2, dx1, dx2])
        return x_save

    # linearize system about its orbit
    t0 = 0.0
    dt = 0.001
    tf = T
    N = int(tf / dt) + 1

    y_euler = euler_full(x0_orbit, T, dt)

    J_big = np.zeros((N+1, 4, 4))
    for i in range(N+1):
        J_big[i] = linearize_system(y_euler[i])


    def euler_linear(x0, T, dt):
        # x0 = x0 - y_euler[0]

        x1 = x0[0]
        x2 = x0[1]
        dx1 = x0[2]
        dx2 = x0[3]

        N = int(T / dt) + 1
        x_save = np.zeros((N+1, 4))
        x_save[0, :] = x0

        for i in range(N):
            x_current = np.array([x1, x2, dx1, dx2])
            x_dot = J_big[i] @ (x_current)

            x1_dot = x_dot[0]
            x2_dot = x_dot[1]
            dx1_dot = x_dot[2] 
            dx2_dot = x_dot[3] 

            x1 += x1_dot * dt
            x2 += x2_dot * dt
            dx1 += dx1_dot * dt
            dx2 += dx2_dot * dt

            x_save[i+1, :] = np.array([x1, x2, dx1, dx2])
        return x_save

    perturbation = 1e-2 * np.eye(4)

    #X_0 = np.zeros((4, 4))
    X_0 = perturbation
    #for i in range(4):
    #    X_0[:, i] = x0_orbit + perturbation[:, i]

    # sim 1:
    x01 = X_0[:, 0]
    x_sim1 = euler_linear(x01, T, dt) # + y_euler
    x1 = x_sim1[-1, :]

    # fig, ax = plt.subplots()
    # ax.plot(x_sim1[:, 0], '--')
    # ax.plot(y_euler[:, 0], '-')
    # ax.plot(x_sim1[:, 1], '--')
    # ax.plot(y_euler[:, 1], '-')
    # ax.plot(x_sim1[:, 2], '--')
    # ax.plot(y_euler[:, 2], '-')
    # ax.plot(x_sim1[:, 3], '--')
    # ax.plot(y_euler[:, 3], '-')
    # plt.show()


    # sim 2:
    x02 = X_0[:, 1]
    x_sim2 = euler_linear(x02, T, dt) #+ y_euler
    x2 = x_sim2[-1, :]

    # sim 3:
    x03 = X_0[:, 2]
    x_sim3 = euler_linear(x03, T, dt) #+ y_euler
    x3 = x_sim3[-1, :]

    # sim 4:
    x04 = X_0[:, 3]
    x_sim4 = euler_linear(x04, T, dt) #+ y_euler
    x4 = x_sim4[-1, :]

    X_T = np.column_stack((x1, x2, x3, x4))
    monodromy_matrix = np.linalg.inv(X_0) @ X_T # np.linalg.solve(X_0, X_T)
    mult, _ = np.linalg.eig(monodromy_matrix)

    return mult




if __name__ == "__main__":
    # example parameters
    kappa_5_con = 50.0
    Omega_con = 0.5
    xa_con = 0.2
    ra_con = 0.3
    mu_con = 0.8
    a_con = -0.3
    theta_con = 0
    mbar = 12
    k_3 = -1

    T = 5.0  # example period
    x0_orbit = np.array([0.1, 0.1, 0.1, 0.1])  # example state on the orbit

    f = get_floquet(kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con, mbar, k_3, T, x0_orbit)
    print("Floquet Multipliers:", np.abs(f))



