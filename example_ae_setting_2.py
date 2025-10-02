import numpy as np

class ae_set():
    def __init__(self, kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con):
        self.kappa_5_con = kappa_5_con
        self.Omega_con = Omega_con
        self.ra_con = ra_con
        self.xa_con = xa_con
        self.mu_con = mu_con
        self.a_con = a_con
        self.theta_con = 0
    # -------------
    # Optimization
    # -------------
    # def obj1(self, w, x):
    #     """
    #     Objective function 1 - Maximize real(lambda)
    #     """
    #     pass

    # def obj(self, w, x):
    #     """
    #     Objective function 1 - Minimize composite mass and stiffness
    #     """
    #     mbar, kappa_3 = _extractDVs(x)
    #     obj = mbar - kappa_3 ** 2

    #     return obj

    # def pobj_px(w, x):
    #     """
    #     pobj / px
    #     """
    #     mbar, kappa_3 = _extractDVs(x)
    #     pobj_px = np.zeros(2)

    #     # mbar
    #     pobj_px[0] = 1.0
    #     # kappa_3
    #     pobj_px[1] = - 2 * kappa_3

    #     return pobj_px


    # def pobj_pw(w, x):
    #     """
    #     pobj / pw
    #     """
    #     return np.zeros(4)


    # -------------
    # Bottom level
    # -------------

    # Private convenience functions
    def _extractDVs(self, x):
        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]
        return mbar, kappa_3

    def _extractState(self, w):
        # Extract state var
        hbar = w[0]
        alpha = w[1]
        hbar_dot = w[2]
        alpha_dot = w[3]
        return hbar, alpha, hbar_dot, alpha_dot

    def res(self, w, x, theta):
        """
        Residual / forcing term
        """

        # Compute A
        A = self.evalA(x)

        # Nonlinear force
        Fnl = self.evalFnl(w, x, theta)

        # Total residual/force
        res = np.zeros(4)
        res[:] += A.dot(w) + Fnl[:]

        return res

    def pres_pw(self, w, x):
        """
        pres / pw
        """

        # Compute A
        A = self.evalA(x)

        # Compute
        pFnlpw = self.evalpFnlpw(w, x)

        # Total residual/force
        pres_pw = np.zeros((4, 4))
        pres_pw[:, :] += A[:, :]
        pres_pw[:, :] += pFnlpw[:, :]

        return pres_pw

    def pres_px(self, w, x):
        """
        pres / px
        """

        pA_pmbar, pA_pkappa_3 = self.evalpApx(w, x)

        pFnl_px = self.evalpFnlpx(w, x)

        # Assemble residual
        pres_px = np.zeros((4, 2))
        # pres / pmbar
        pres_px[:, 0] += pA_pmbar.dot(w) + pFnl_px[:,0]
        # pres / p kappa_3
        pres_px[:, 1] += pFnl_px[:,1]

        return pres_px

    def evalFnl(self, w, x, theta):
        # Get DVs and states
        mbar, kappa_3 = self._extractDVs(x)
        hbar, alpha, hbar_dot, alpha_dot = self._extractState(w)

        # Nonlinear force
        v = np.zeros(2)
        v[1] = self.ra_con ** 2 * (kappa_3 * (alpha + theta) ** 3 + self.kappa_5_con * (alpha + theta) ** 5)
        MInv = self.evalMInv(x)

        Fnl = np.zeros(4)
        Fnl[2:] = -1.0 * np.dot(MInv, v)

        return Fnl

    def evalpFnlpw(self, w, x):
        """
        pFnl / pw
        """

        # Get DVs and states
        mbar, kappa_3 = self._extractDVs(x)
        hbar, alpha, hbar_dot, alpha_dot = self._extractState(w)

        # Nonlinear force
        # pfnl/pw = p/pw(- M^{-1} * v) = -[pM{-1}/pw * v + M^{-1} * pv/pw] = -M^{-1} * pv/pw
        MInv = self.evalMInv(x)
        pv_pw = np.zeros((2,4))
        # only have alpha (second variable in w) and only last (second) entry in v is non-zero, everything else is zero
        pv_pw[1,1] = self.ra_con ** 2 * (kappa_3 * 3.0 * (alpha + self.theta_con) ** 2 + self.kappa_5_con * 5.0 * (alpha + self.theta_con) ** 4)

        pFnlpw = np.zeros((4, 4))
        pFnlpw[2:, :] = -1.0 * np.dot(MInv, pv_pw)

        return pFnlpw

    def evalpFnlpx(self, w, x):
        """
        pFnl / px
        """

        mbar, kappa_3 = self._extractDVs(x)
        hbar, alpha, hbar_dot, alpha_dot = self._extractState(w)

        # Nonlinear force
        # pfnl/px = p/px(-M^{-1} * v) = -[pM{-1}/px * v + M^{-1} * pv/px]

        # p/pmbar(-M^{-1} * v) = -[pM{-1}/pmbar * v + M^{-1} * pv/pmbar] = - pM{-1}/pmbar * v
        pMInv_pmbar, MInv_pkappa_3 = self.evalpMInvpx(x)
        v = np.zeros(2)
        v[1] = self.ra_con ** 2 * (kappa_3 * (alpha + self.theta_con) ** 3 + self.kappa_5_con * (alpha + self.theta_con) ** 5)
        pFnl_pmbar = -1.0 * np.dot(pMInv_pmbar, v)

        # p/pkappa_3(-M^{-1} * v) = -[pM{-1}/pkappa_3 * v + M^{-1} * pv/pkappa_3] = - M^{-1} * pv/pkappa_3
        MInv = self.evalMInv(x)
        pv_pkappa_3 = np.zeros(2)
        pv_pkappa_3[1] = self.ra_con ** 2 * (alpha + self.theta_con) ** 3
        pFnl_pkappa_3 = -1.0 * np.dot(MInv, pv_pkappa_3)

        pFnl_px = np.zeros((4,2))
        pFnl_px[2:,0] = pFnl_pmbar
        pFnl_px[2:,1] = pFnl_pkappa_3

        return pFnl_px

    def evalA(self, x):
        """
        Linear coefficient matrix A
        """

        mbar, kappa_3 = self._extractDVs(x)

        # Linear force (Identity matrix upper right)
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0

        # M^{-1} * K
        MInv = self.evalMInv(x)
        K = self.evalK(x)
        MInv_K = np.dot(MInv, K)

        # M^{-1} * Da
        Da = self.evalDa(x)
        MInv_Da = np.dot(MInv, Da)

        A[2:4, 0:2] += -1.0 * MInv_K[:, :]
        A[2:4, 2:4] += -1.0 * MInv_Da[:, :]

        return A

    def evalpApx(self, w, x):
        """
        pA/px
        """

        # mbar
        pA_pmbar = np.zeros((4, 4))

        # p/px(M^{-1} * K) = pM{-1}/px * K + M^{-1} * pK/px
        pMInv_pmbar, pMInv_pkappa_3 = self.evalpMInvpx(x)
        K = self.evalK(x)
        MInv = self.evalMInv(x)
        pK_pmbar = self.evalpKpx(x)
        pMinvK_pmbar = np.dot(pMInv_pmbar, K) + np.dot(MInv, pK_pmbar)

        # p/px(M^{-1} * Da) = pM{-1}/px * Da + M^{-1} * pDa/px
        Da = self.evalDa(x)
        pDa_pmbar = self.evalpDapx(x)
        pMinvDa_pmbar = np.dot(pMInv_pmbar, Da) + np.dot(MInv, pDa_pmbar)

        pA_pmbar[2:4, 0:2] += -1.0 * pMinvK_pmbar[:, :]
        pA_pmbar[2:4, 2:4] += -1.0 * pMinvDa_pmbar[:, :]

        # A is not a function of kappa_3 so partial derivative is zero
        pA_pkappa_3 = np.zeros((4, 4))

        return pA_pmbar, pA_pkappa_3

    # ----- Mass Matrices

    def evalMs(self):
        Ms = np.zeros((2,2))
        Ms[0,0] = 1.0
        Ms[0,1] = self.xa_con
        Ms[1,0] = self.xa_con
        Ms[1,1] = self.ra_con ** 2
        return Ms

    def evalMa(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        Ma = np.zeros((2,2))
        Ma[0,0] = 1.0
        Ma[0,1] = -self.a_con
        Ma[1,0] = -self.a_con
        Ma[1,1] = 1.0/8.0 + self.a_con ** 2
        coeff = 1.0/mbar
        Ma = coeff * Ma
        return Ma

    def evalM(self, x):
        return self.evalMs() + self.evalMa(x)

    # def evalpMpx(x):
    #     """
    #     pM/px = p/px(Ms + Ma) = pMa/px
    #     """
    #     return evalpMapx(x)

    # def evalpMapx(x):
    #     mbar, kappa_3 = _extractDVs(x)

    #     # only need mbar (wrt kappa_3 will be zero)
    #     Ma = evalMa(x)
    #     pMa_pmbar = -1.0/mbar * Ma
    #     return pMa_pmbar

    def evalMInv(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        tmp = 1.0/8.0 + self.a_con ** 2
        det = self.ra_con**2 + tmp/mbar + self.ra_con**2/mbar + tmp/mbar**2 - self.xa_con**2 + 2*self.a_con*self.xa_con/mbar - (self.a_con/mbar)**2

        MInv = np.zeros((2,2))
        MInv[0,0] = self.ra_con** 2 + tmp/mbar
        MInv[0,1] = -(self.xa_con - self.a_con/mbar)
        MInv[1,0] = -(self.xa_con - self.a_con/mbar)
        MInv[1,1] = 1.0 + 1.0/mbar
        MInv = 1.0 / det * MInv

        return MInv

    def evalpMInvpx(self, x):
        """
        pMInv/px = p(1/c(x) * B(x)) / px = -1/c^2 * pc/px * B + 1/c pB/px
        """
        mbar, kappa_3 = self._extractDVs(x)

        tmp = 1.0/8.0 + self.a_con ** 2
        det = self.ra_con**2 + tmp/mbar + self.ra_con**2/mbar + tmp/mbar**2 - self.xa_con**2 + 2*self.a_con*self.xa_con/mbar - (self.a_con/mbar)**2
        oneOverDet = 1.0 / det

        # mbar only (not a dependence on kappa_3)
        # pc/pmbar
        ddet_dmbar = - tmp/mbar**2 - self.ra_con**2/mbar**2 - 2*tmp/mbar**3 - 2*self.a_con*self.xa_con/mbar**2 + 2*self.a_con**2/mbar**3
        Minv = self.evalMInv(x)

        # pB/pmbar
        pB_pmbar = np.zeros((2,2))
        pB_pmbar[0,0] = -tmp/mbar**2
        pB_pmbar[0,1] = -self.a_con/mbar**2
        pB_pmbar[1,0] = -self.a_con/mbar**2
        pB_pmbar[1,1] = -1.0/mbar**2

        MInv_pmbar = -oneOverDet * ddet_dmbar * Minv + oneOverDet * pB_pmbar

        MInv_pkappa_3 = np.zeros((2,2))

        return MInv_pmbar, MInv_pkappa_3


    # ----- Stiffness Matrices
    def evalDa(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        Da = np.zeros((2,2))
        Da[0,0] = 1.0
        Da[0,1] = 1.0 - self.a_con
        Da[1,0] = -(1.0/2.0 + self.a_con)
        Da[1,1] = self.a_con * (self.a_con - 1.0/2.0)

        coeff = 2*self.mu_con/mbar
        Da = coeff * Da

        return Da

    def evalpDapx(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        # only need mbar (wrt kappa_3 will be zero)
        Da = self.evalDa(x)
        pDa_pmbar = -1.0/mbar * Da
        return pDa_pmbar


    # ----- Stiffness Matrices
    def evalKs(self):
        Ks = np.zeros((2,2))
        Ks[0,0] = self.Omega_con ** 2    
        Ks[1,1] = self.ra_con ** 2

        return Ks

    def evalKa(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        Ka = np.zeros((2,2))
        Ka[0,1] = 1.0
        Ka[1,1] = -(1.0/2.0 + self.a_con)

        coeff = 2*self.mu_con**2/mbar
        Ka = coeff * Ka

        return Ka

    def evalK(self, x):
        return self.evalKs() + self.evalKa(x)

    def evalpKpx(self, x):
        """
        pK/px = p/px(Ks + Ka) = pKa/px
        """
        return self.evalpKapx(x)

    def evalpKapx(self, x):
        mbar, kappa_3 = self._extractDVs(x)

        # only need mbar (wrt kappa_3 will be zero)
        Ka = self.evalKa(x)
        pKa_pmbar = -1.0/mbar * Ka
        return pKa_pmbar

    # ----------
    # Top level
    # ----------

    def f_LST(self, v, x):

        """
        Linear stability function.
        """
        # Here we return the real part of the eigenvalue
        return v[-2]

    def f_pLST_pv(self, v, x):
        """
        pLST / pv
        """

        ndof = np.shape(v)[0]
        pLST_pv = np.zeros(ndof)
        pLST_pv[-2] = 1.0

        return pLST_pv
