# This files contains LCO problem and its time-spectral solver.

import numpy as np
from scipy import optimize
from TS import timeder_mat
import copy

def fft_interp(coeffs, phase):
    # 1D FFT interpolate for arbitrary x
    # assume x is periodic on [0,1] interval

    # Assume odd number
    n_freq = (len(coeffs) - 1) // 2
    scale = len(coeffs)

    val = coeffs[0]
    for i in range(n_freq):
        val += coeffs[i + 1] * np.exp(1j * phase * (i + 1))
        val += coeffs[- i - 1] * np.exp(- 1j * phase * (i + 1))

    val *= 1/scale
    return np.real(val)
class LCO_TS_stencil(object):

    """
        A stencil made of three LCO points. It is used to
        generate the fitted bifurcation diagram.
        The points from the stencil are assumed to be solved
        already.
    """

    def __init__(self, LCO_TS_list, h_arr):

        # An array of prescribed motion in the order of
        # h-, h0, h+
        self.h_arr = copy.deepcopy(h_arr)

        # A list of LCO objects in the order of
        # h-, h0, h+
        self.LCO_TS_list = LCO_TS_list

        # Compute der
        self.nx = self.LCO_TS_list[0].get_nx()
        self.dmu_arr_dx = np.zeros((3, self.nx))
        for i in range(3):
            dmudx = copy.deepcopy(self.LCO_TS_list[i].solve_bot_dmudx())
            self.dmu_arr_dx[i, :] = dmudx[:]

        # An array of parameters
        self.mu_arr = np.zeros(3)
        for i in range(3):
            self.mu_arr[i] = self.LCO_TS_list[i].get_state_var()[0]

        # An array of fitting coef
        # a h^3 + b h^2 + c h + d = mu
        # Since c=0, we ignore it.
        # coeff = [a, b, d]
        self.coeff = np.zeros(3)

        # M_coeff_mu * coeff (w/o c) = mu_arr
        self.M_coeff_mu = np.zeros((3, 3))
        for i in range(3):
            h = self.h_arr[i]
            self.M_coeff_mu[i, 0] = h ** 3
            self.M_coeff_mu[i, 1] = h ** 2
            self.M_coeff_mu[i, 2] = 1.0

        # Stability measure
        self.stab_measure = 0.0

    def solve(self, xin_in_list):

        xin_out_list = []
        for i in range(3):
            xin_in = xin_in_list[i]
            xin_out = self.LCO_TS_list[i].solve(xin_in)

            xin_out_list.append(xin_out)

        return xin_out_list

    def get_state_var(self):

        sol_list = []
        for i in range(3):
            sol = self.LCO_TS_list[i].get_state_var()
            sol_list.append(sol)

        return sol_list

    def get_mu(self):

        return self.mu_arr[1]

    def compute_mu_der(self):

        return self.LCO_TS_list[1].solve_bot_dmudx()

    def reset_mu(self):

        for i in range(3):
            self.mu_arr[i] = self.LCO_TS_list[i].get_state_var()[0]

    def compute_abd(self):

        self.coeff[:] = np.linalg.solve(self.M_coeff_mu, self.mu_arr)

    def get_abd(self):

        return self.coeff

    def compute_stability(self):

        self.compute_abd()

        a = self.coeff[0]
        b = self.coeff[1]
        h0 = self.h_arr[1]

        self.stab_measure = 3.0 * a * h0 ** 2 + 2.0 * b * h0

    def get_stability(self):

        return self.stab_measure

    def compute_stability_der(self):

        h0 = self.h_arr[1]

        # d stability_measure / d coeff
        # stability_measure = 3 a h0^2 + 2 b h0
        dstab_dcoeff = np.zeros(3)
        dstab_dcoeff[0] = 3 * h0 ** 2
        dstab_dcoeff[1] = 2 * h0

        # d coeff / d mu_arr
        dcoeff_dmu_arr = np.linalg.inv(self.M_coeff_mu)

        # Solve for intermediate adjoint
        for i in range(3):
            dmudx = copy.deepcopy(self.LCO_TS_list[i].solve_bot_dmudx())
            self.dmu_arr_dx[i, :] = dmudx[:]

        # d stability_measure / dx = d stability_measure / d coeff
        #                          * d coeff / d mu_arr
        #                          * d mu_arr / dx
        self.dstab_dx = (dstab_dcoeff.dot(dcoeff_dmu_arr)).dot(self.dmu_arr_dx)

    def get_stability_der(self):

        return self.dstab_dx

    def set_design_var(self, x):

        for i in range(3):

            self.LCO_TS_list[i].set_design_var(x)

    def evaluate(self, h):

        return self.coeff[0] * h ** 3 + self.coeff[1] * h ** 2 + self.coeff[2]


class LCO_TS(object):

    """
        LCO class. It has the following capabilities:
        1. The LCO problem dynamic description.
        2. The TS JFNK LCO solver.
    """

    def __init__(
        self,
        force,
        ntimeinstance,
        ndof,
        x=None,
        pforcepx_func=None,
        pforcepw_func=None,
        pforcepmu_func=None,
        pforcepomega_func=None,
    ):

        """
            Inputs:
            1. "force" is a function.
            dot{u} - force(mu, w) = 0.
            where dot is time der, mu bifurcation parameter, 
            and w state variable.
            2. ntimeinstance is time instance.
            3. ndof is problem size for one time instance.
        """

        self.force = force

        self.ntimeinstance = ntimeinstance
        self.ndof = ndof

        self.w = np.zeros(ntimeinstance * ndof)

        if x is not None:
            self.x = x
            self.nx = len(x)
        else:
            self.nx = 0
        if pforcepx_func is not None:
            self.pforcepx_func = pforcepx_func
        if pforcepw_func is not None:
            self.pforcepw_func = pforcepw_func
        if pforcepmu_func is not None:
            self.pforcepmu_func = pforcepmu_func
        if pforcepomega_func is not None:
            self.pforcepomega_func = pforcepomega_func

        # set the permutation matrix for time derivative
        self._set_permutation()

    # ============================
    # Primal analysis
    # ============================

    def generate_xin(self):

        """
            Generate an initial guess, xin.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof
        mag = self.mag_0
        pha = self.pha_0

        mu = 0.7
        T = 10.0
        omega = 2.0 * np.pi / T

        w0 = np.zeros(ntimeinstance * ndof)
        # for i in range(ntimeinstance * ndof):
        #     if (i % ndof == 0):
        #         # Only specify the first DOF
        #         w0[i] = mag * np.sin(float(i // ndof) \
        #             / float(ntimeinstance) * 2.0 * np.pi + pha)
        #     if (i % ndof == 2):
        #         # Only specify the first DOF
        #         w0[i] = omega * mag * np.cos(float(i // ndof) \
        #             / float(ntimeinstance) * 2.0 * np.pi + pha)
        for i in range(ntimeinstance * ndof):
            w0[i] = mag * np.sin(float(i) / float(ntimeinstance * ndof) * 2.0 * np.pi + pha)

        xin = np.zeros(ntimeinstance * ndof + 2)
        xin[0] = mu
        xin[1] = omega
        xin[2:] = w0[:]

        return xin

    def set_design_var(self, x):

        """
            Set the design variables.
        """

        self.x = copy.deepcopy(x)
        self.nx = len(x)

    def set_motion_mag_pha(self, mag, pha, ind_p=0):

        """
            Set motion magnitude and residual.
        """

        self.mag_0 = mag
        self.pha_0 = pha

        self.ind_p = ind_p

    def _set_permutation(self):

        """
            The permutation matrix used to compute time derivative.
            PT D P
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        P = np.zeros((ndof * ntimeinstance, ndof * ntimeinstance))
        for i in range(ndof):
            for j in range(ntimeinstance):
                P[i * ntimeinstance + j, j * ndof + i] = 1.0
        PT = np.transpose(P)

        self.P = P
        self.PT = PT

    def res_wrapper(self, u):

        """
            Wrapper for "set_state_var" and 
            "compute_res".
        """

        self.set_state_var(u)
        res = self.compute_res()

        return res

    def set_state_var(self, u):

        """
            Set state var.
            u = [mu, omega, w_1, ..., w_N].
        """

        self.mu = u[0]
        self.omega = u[1]
        self.w = u[2:]

    def get_state_var(self):

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        u = np.zeros(ntimeinstance * ndof + 2)

        u[0] = self.mu
        u[1] = self.omega
        u[2:] = self.w[:]

        return u

    def compute_time_der(self):

        """
            Compute the time derivative using spectral differentiation.
        """

        omega = self.omega
        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        T = 2.0 * np.pi / omega
        D = timeder_mat(T, ntimeinstance)

        w_der = np.zeros(ntimeinstance * ndof)

        for i in range(ndof):
            wdof = np.zeros(ntimeinstance)
            for j in range(ntimeinstance):
                wdof[j] = self.w[j * ndof + i]

            wdof_der = D.dot(wdof)

            for j in range(ntimeinstance):
                w_der[j * ndof + i] = wdof_der[j]

        return w_der

    def compute_dyn_res(self):

        """
            Compute the dynamic residual.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        mu = self.mu
        w = self.w

        force = self.force

        w_der = self.compute_time_der()

        res_dyn = np.zeros(ntimeinstance * ndof)

        for i in range(ntimeinstance):

            w_loc = w[i * ndof : (i + 1) * ndof]
            res_dyn[i * ndof : (i + 1) * ndof] = -force(mu, w_loc, self.x)[:]

        res_dyn[:] += w_der[:]

        return res_dyn

    def compute_motion_res(self):

        """
            Compute the prescirbed motion residual.
        """

        # Prescribed motion residual
        # Fourier coefficient

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof
        w = self.w
        ind_p = self.ind_p

        wdof_1 = np.zeros(ntimeinstance)
        for i in range(ntimeinstance):
            wdof_1[i] = w[i * ndof + ind_p]

        c = (1.0 / float(ntimeinstance)) * np.fft.fft(wdof_1)

        c_1r = np.real(c[1])
        c_n1r = np.real(c[-1])
        c_1i = np.imag(c[1])
        c_n1i = np.imag(c[-1])

        Cc = c_1r + c_n1r
        Cs = -c_1i + c_n1i

        mag = np.sqrt(Cc ** 2 + Cs ** 2)
        pha = np.arcsin(Cc / mag)

        res_mag = mag - self.mag_0
        res_pha = pha - self.pha_0

        return res_mag, res_pha
    
    def fft(self):

        ndof = self.ndof
        ntimeinstance = self.ntimeinstance

        coeff_glo = np.zeros((ndof, ntimeinstance), dtype=np.complex128)

        for i in range(ndof):
            coeff = np.zeros(ntimeinstance)
            for j in range(ntimeinstance):
                coeff[j] = self.w[j * ndof + i]

            coeff = np.fft.fft(coeff)

            coeff_glo[i, :] = coeff
            
        self.coeff_glo = coeff_glo



    def interp(self, t):

        omega = self.omega
        ndof = self.ndof
        ntimeinstance = self.ntimeinstance

        coeff_glo = self.coeff_glo

        T = 2.0 * np.pi / omega
        pha = t / T * (2*np.pi)

        w_interp = np.zeros(ndof)
        for i in range(ndof):
            coeff = coeff_glo[i, :]
            w_interp[i] = fft_interp(coeff, pha)

        return w_interp



    def compute_res(self):

        """
            Compute residual.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        # prescribed motion residual
        res_mag, res_pha = self.compute_motion_res()

        # Dynamic residual
        res_dyn = self.compute_dyn_res()

        # combine residual
        res = np.zeros(ntimeinstance * ndof + 2)
        res[0] = res_mag
        res[1] = res_pha
        res[2:] = res_dyn[:]

        return res

    def solve(self, xin, tol=1e-3):

        """
            Solve for the LCO state vars.
        """

        # sol = optimize.fsolve(self.res_wrapper, xin)
        sol = optimize.newton_krylov(self.res_wrapper, xin, f_tol=tol)
        

        self.mu = sol[0]
        self.omega = sol[1]
        self.w[:] = sol[2:]

        # print(self.res_wrapper(sol))
        print(self.w[:])

        return sol

    def get_nx(self):

        return self.nx

    # ============================
    # Derivative analysis
    # ============================

    def compute_pRmagphapu(self):

        """
            Compute derivative of the derivative of prescribed motion phase and 
            magnitude with respect to the displacemet.
        """

        # Number of time instances and d.o.f (per time instance)
        ntimeinstance = self.ntimeinstance
        ndof = self.ndof
        ind_p = self.ind_p

        disp = np.zeros(ntimeinstance * ndof)
        disp[:] = self.w[:]

        # Name the 1st dof as alpha
        alpha = np.zeros(ntimeinstance)
        for sps in range(ntimeinstance):
            alpha[sps] = disp[ndof * sps + ind_p]

        coeff_array = np.fft.fft(alpha)
        cp1r = np.real(coeff_array[1])
        cp1i = np.imag(coeff_array[1])
        cm1r = np.real(coeff_array[-1])
        cm1i = np.imag(coeff_array[-1])

        Cc = (cp1r + cm1r) / float(ntimeinstance)
        Cs = (-cp1i + cm1i) / float(ntimeinstance)

        # pcoeff/palpha
        pcp1r_palpha = np.zeros(ntimeinstance)
        pcp1i_palpha = np.zeros(ntimeinstance)
        pcm1r_palpha = np.zeros(ntimeinstance)
        pcm1i_palpha = np.zeros(ntimeinstance)

        delta1 = (2.0 * np.pi) / ntimeinstance
        deltaNm1 = (2.0 * np.pi) * (ntimeinstance - 1.0) / float(ntimeinstance)
        for i in range(ntimeinstance):
            pcp1r_palpha[i] = np.cos(float(i) * delta1)
            pcm1r_palpha[i] = np.cos(float(i) * deltaNm1)
            pcp1i_palpha[i] = (-1.0) * np.sin(float(i) * delta1)
            pcm1i_palpha[i] = (-1.0) * np.sin(float(i) * deltaNm1)

        pcpalpha = np.zeros((4, ntimeinstance))

        pcpalpha[0, :] = pcp1r_palpha[:]
        pcpalpha[1, :] = pcp1i_palpha[:]
        pcpalpha[2, :] = pcm1r_palpha[:]
        pcpalpha[3, :] = pcm1i_palpha[:]

        # pCcs/pcoeff
        pCpc = np.zeros((2, 4))

        pCpc[0, 0] = 1.0 / float(ntimeinstance)
        pCpc[0, 2] = 1.0 / float(ntimeinstance)
        pCpc[1, 1] = -1.0 / float(ntimeinstance)
        pCpc[1, 3] = 1.0 / float(ntimeinstance)

        # pmagpha/pCcs
        sum_2 = Cc ** 2 + Cs ** 2
        sqrt_sum_2 = np.sqrt(sum_2)

        pmagpha_pC = np.zeros((2, 2))
        pmagpha_pC[0, 0] = Cc / sqrt_sum_2
        pmagpha_pC[0, 1] = Cs / sqrt_sum_2
        pmagpha_pC[1, 0] = abs(Cs) / sum_2
        pmagpha_pC[1, 1] = -Cc * np.sign(Cs) / sum_2

        # Jacobian
        pRmagpha_palpha = np.dot(np.dot(pmagpha_pC, pCpc), pcpalpha)
        pRmag_palpha = pRmagpha_palpha[0, :]
        pRpha_palpha = pRmagpha_palpha[1, :]

        # fill in...
        pRmag_pu = np.zeros(ndof * ntimeinstance)
        pRpha_pu = np.zeros(ndof * ntimeinstance)
        for sps in range(ntimeinstance):
            pRmag_pu[ind_p + ndof * sps] = pRmag_palpha[sps]
            pRpha_pu[ind_p + ndof * sps] = pRpha_palpha[sps]

        pRpmagpha_pu = np.zeros((2, ndof * ntimeinstance))
        pRpmagpha_pu[0, :] = pRmag_pu[:]
        pRpmagpha_pu[1, :] = pRpha_pu[:]

        return pRpmagpha_pu

    def compute_pSpomega(self):

        """
            Compute p R / p omega which is equal to
                (p(P^T D P) / pomega) w.
            We use the fact that this number is proportional to 
            omega^2, i.e., D a omega^2. Thus, 
                (p(P^T D P) / pomega) w = (2 / omega) P^T D P w
        """

        # Take in variables
        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        omega = self.omega

        # Form time der matrix
        T = 2.0 * np.pi / omega
        D_oneDof = timeder_mat(T, ntimeinstance)

        D = np.zeros((ntimeinstance * ndof, ntimeinstance * ndof))
        for i in range(ndof):
            D[i * ntimeinstance : (i + 1) * ntimeinstance, i * ntimeinstance : (i + 1) * ntimeinstance] = D_oneDof[:, :]

        # Permutation
        D = (self.PT.dot(D)).dot(self.P)

        # Result
        v = D.dot(self.w)
        v *= 2.0 / omega

        return v

    def compute_pSpmu(self):

        """
            Compute p R / p mu. This is problem specific and the user shall 
            provide this information.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        mu = self.mu

        pRpmu = np.zeros(ntimeinstance * ndof)
        w_loc = np.zeros(ndof)
        for i in range(ntimeinstance):

            w_loc[:] = self.w[i * ndof : (i + 1) * ndof]

            pRpmu[i * ndof : (i + 1) * ndof] += -self.pforcepmu_func(mu, w_loc, self.x)

        return pRpmu

    def compute_pSpw(self):

        """
            Compute p R / p w. This is problem specific and the user shall 
            provide this information.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        mu = self.mu
        omega = self.omega

        # Jacobian
        pRpw = np.zeros((ntimeinstance * ndof, ntimeinstance * ndof))

        # Spatial component
        w_loc = np.zeros(ndof)
        for i in range(ntimeinstance):

            w_loc[:] = self.w[i * ndof : (i + 1) * ndof]

            pRpw[i * ndof : (i + 1) * ndof, i * ndof : (i + 1) * ndof] += -self.pforcepw_func(mu, w_loc, self.x)

        # Temporal component
        T = 2.0 * np.pi / omega
        D_oneDof = timeder_mat(T, ntimeinstance)

        D = np.zeros((ntimeinstance * ndof, ntimeinstance * ndof))
        for i in range(ndof):
            D[i * ntimeinstance : (i + 1) * ntimeinstance, i * ntimeinstance : (i + 1) * ntimeinstance] = D_oneDof[:, :]

        D = (self.PT.dot(D)).dot(self.P)

        pRpw += D

        return pRpw

    def compute_bot_pRpx(self):

        """
            pR / px for the bottom LCO system.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof
        nx = self.nx

        mu = self.mu

        # Compute individual components
        # p R_m / p x, p R_p / p x
        pRmagphapx = np.zeros((2, nx))

        self.pRpx = np.zeros((ndof * ntimeinstance + 2, nx))
        self.pRpx[0:2, :] = pRmagphapx[:, :]

        # p S / p x
        w_loc = np.zeros(ndof)
        for i in range(ntimeinstance):

            # Retrieve the state variables
            w_loc[:] = self.w[i * ndof : (i + 1) * ndof]

            # Evaluate the partial derivative
            pSpx = -self.pforcepx_func(mu, w_loc, self.x)

            # Fill in
            self.pRpx[2 + i * ndof : 2 + (i + 1) * ndof, :] = pSpx[:, :]

    def compute_bot_pRpq(self):

        """
            pR / pq for the bottom LCO system.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        # Compute individual components
        # p R_m / p w, p R_p / p w
        pRmagphapu = self.compute_pRmagphapu()

        # p S / p mu
        pSpmu = self.compute_pSpmu()

        # p S / p omega
        pSpomega = self.compute_pSpomega()

        # p S / p w
        pSpw = self.compute_pSpw()

        # Fill in the entries
        self.pRpq = np.zeros((ntimeinstance * ndof + 2, ntimeinstance * ndof + 2))

        self.pRpq[0, 2:] = pRmagphapu[0, :]
        self.pRpq[1, 2:] = pRmagphapu[1, :]

        self.pRpq[2:, 0] = pSpmu[:]
        self.pRpq[2:, 1] = pSpomega[:]

        self.pRpq[2:, 2:] = pSpw[:, :]

        return self.pRpq

    def solve_bot_dmudx(self):

        """
            Compute 
                d mu / dx,
            for LCO. Mainly for debugging purpose.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof
        nx = self.nx

        # Set up pf / px and pf / pq
        pfpx = np.zeros(nx)

        # Set up pf / pq
        pfpq = np.zeros(ndof * ntimeinstance + 2)
        pfpq[0] = 1.0

        # Set up p R / p x
        self.compute_bot_pRpx()

        # Set up p R / p q
        self.compute_bot_pRpq()

        # Solve for adjoint
        psi = np.linalg.solve(self.pRpq.T, pfpq)

        # Compute the total der
        dfdx = pfpx - psi.dot(self.pRpx)

        return dfdx

    # ============================
    # Postprocess
    # ============================

    def gen_bif_diag(self, xinmin, magmin, magmax, N):

        """
            Generate the bifurcation diagram.
        """

        ntimeinstance = self.ntimeinstance
        ndof = self.ndof

        xin = np.zeros(ntimeinstance * ndof + 2)
        xin = xinmin[:]

        mag_arr = np.linspace(magmin, magmax, N)
        mu_arr = np.zeros(N)
        mu_arr[0] = xin[0]

        for i in range(N):

            print("Running the ", i, "-th magnitude from the bifurcation diagram.")

            # Set the magnitude
            mag = mag_arr[i]
            self.set_motion_mag_pha(mag, self.pha_0, ind_p=self.ind_p)

            # Solve for the state
            xin = self.solve(xin)

            # Set current solution for the initial guess for
            # the next point.
            self.set_state_var(xin)

            # Extract LCO index
            mu = xin[0]
            mu_arr[i] = mu

        return mag_arr, mu_arr
