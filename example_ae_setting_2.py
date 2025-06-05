import numpy as np
from example_ae_constants import kappa_5_con, Omega_con, ra_con, xa_con, mu_con, a_con, theta_con


# -------------
# Optimization
# -------------
def obj1(w, x):
    """
    Objective function 1 - Maximize real(lambda)
    """
    pass

def obj(w, x):
    """
    Objective function 1 - Minimize composite mass and stiffness
    """
    mbar, kappa_3 = _extractDVs(x)
    obj = mbar - kappa_3 ** 2

    return obj

def pobj_px(w, x):
    """
    pobj / px
    """
    mbar, kappa_3 = _extractDVs(x)
    pobj_px = np.zeros(2)

    # mbar
    pobj_px[0] = 1.0
    # kappa_3
    pobj_px[1] = - 2 * kappa_3

    return pobj_px


def pobj_pw(w, x):
    """
    pobj / pw
    """
    return np.zeros(4)


# -------------
# Bottom level
# -------------

# Private convenience functions
def _extractDVs(x):
    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]
    return mbar, kappa_3

def _extractState(w):
    # Extract state var
    hbar = w[0]
    alpha = w[1]
    hbar_dot = w[2]
    alpha_dot = w[3]
    return hbar, alpha, hbar_dot, alpha_dot

def res(w, x, theta):
    """
    Residual / forcing term
    """

    # Compute A
    A = evalA(x)

    # Nonlinear force
    Fnl = evalFnl(w, x, theta)

    # Total residual/force
    res = np.zeros(4)
    res[:] += A.dot(w) + Fnl[:]

    return res

def pres_pw(w, x):
    """
    pres / pw
    """

    # Compute A
    A = evalA(x)

    # Compute
    pFnlpw = evalpFnlpw(w, x)

    # Total residual/force
    pres_pw = np.zeros((4, 4))
    pres_pw[:, :] += A[:, :]
    pres_pw[:, :] += pFnlpw[:, :]

    return pres_pw

def pres_px(w, x):
    """
    pres / px
    """

    pA_pmbar, pA_pkappa_3 = evalpApx(w, x)

    pFnl_px = evalpFnlpx(w, x)

    # Assemble residual
    pres_px = np.zeros((4, 2))
    # pres / pmbar
    pres_px[:, 0] += pA_pmbar.dot(w) + pFnl_px[:,0]
    # pres / p kappa_3
    pres_px[:, 1] += pFnl_px[:,1]

    return pres_px

def evalFnl(w, x, theta):
    # Get DVs and states
    mbar, kappa_3 = _extractDVs(x)
    hbar, alpha, hbar_dot, alpha_dot = _extractState(w)

    # Nonlinear force
    v = np.zeros(2)
    v[1] = ra_con ** 2 * (kappa_3 * (alpha + theta) ** 3 + kappa_5_con * (alpha + theta) ** 5)
    MInv = evalMInv(x)

    Fnl = np.zeros(4)
    Fnl[2:] = -1.0 * np.dot(MInv, v)

    return Fnl

def evalpFnlpw(w, x):
    """
    pFnl / pw
    """

    # Get DVs and states
    mbar, kappa_3 = _extractDVs(x)
    hbar, alpha, hbar_dot, alpha_dot = _extractState(w)

    # Nonlinear force
    # pfnl/pw = p/pw(- M^{-1} * v) = -[pM{-1}/pw * v + M^{-1} * pv/pw] = -M^{-1} * pv/pw
    MInv = evalMInv(x)
    pv_pw = np.zeros((2,4))
    # only have alpha (second variable in w) and only last (second) entry in v is non-zero, everything else is zero
    pv_pw[1,1] = ra_con ** 2 * (kappa_3 * 3.0 * (alpha + theta_con) ** 2 + kappa_5_con * 5.0 * (alpha + theta_con) ** 4)

    pFnlpw = np.zeros((4, 4))
    pFnlpw[2:, :] = -1.0 * np.dot(MInv, pv_pw)

    return pFnlpw

def evalpFnlpx(w, x):
    """
    pFnl / px
    """

    mbar, kappa_3 = _extractDVs(x)
    hbar, alpha, hbar_dot, alpha_dot = _extractState(w)

    # Nonlinear force
    # pfnl/px = p/px(-M^{-1} * v) = -[pM{-1}/px * v + M^{-1} * pv/px]

    # p/pmbar(-M^{-1} * v) = -[pM{-1}/pmbar * v + M^{-1} * pv/pmbar] = - pM{-1}/pmbar * v
    pMInv_pmbar, MInv_pkappa_3 = evalpMInvpx(x)
    v = np.zeros(2)
    v[1] = ra_con ** 2 * (kappa_3 * (alpha + theta_con) ** 3 + kappa_5_con * (alpha + theta_con) ** 5)
    pFnl_pmbar = -1.0 * np.dot(pMInv_pmbar, v)

    # p/pkappa_3(-M^{-1} * v) = -[pM{-1}/pkappa_3 * v + M^{-1} * pv/pkappa_3] = - M^{-1} * pv/pkappa_3
    MInv = evalMInv(x)
    pv_pkappa_3 = np.zeros(2)
    pv_pkappa_3[1] = ra_con ** 2 * (alpha + theta_con) ** 3
    pFnl_pkappa_3 = -1.0 * np.dot(MInv, pv_pkappa_3)

    pFnl_px = np.zeros((4,2))
    pFnl_px[2:,0] = pFnl_pmbar
    pFnl_px[2:,1] = pFnl_pkappa_3

    return pFnl_px

def evalA(x):
    """
    Linear coefficient matrix A
    """

    mbar, kappa_3 = _extractDVs(x)

    # Linear force (Identity matrix upper right)
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    # M^{-1} * K
    MInv = evalMInv(x)
    K = evalK(x)
    MInv_K = np.dot(MInv, K)

    # M^{-1} * Da
    Da = evalDa(x)
    MInv_Da = np.dot(MInv, Da)

    A[2:4, 0:2] += -1.0 * MInv_K[:, :]
    A[2:4, 2:4] += -1.0 * MInv_Da[:, :]

    return A

def evalpApx(w, x):
    """
    pA/px
    """

    # mbar
    pA_pmbar = np.zeros((4, 4))

    # p/px(M^{-1} * K) = pM{-1}/px * K + M^{-1} * pK/px
    pMInv_pmbar, pMInv_pkappa_3 = evalpMInvpx(x)
    K = evalK(x)
    MInv = evalMInv(x)
    pK_pmbar = evalpKpx(x)
    pMinvK_pmbar = np.dot(pMInv_pmbar, K) + np.dot(MInv, pK_pmbar)

    # p/px(M^{-1} * Da) = pM{-1}/px * Da + M^{-1} * pDa/px
    Da = evalDa(x)
    pDa_pmbar = evalpDapx(x)
    pMinvDa_pmbar = np.dot(pMInv_pmbar, Da) + np.dot(MInv, pDa_pmbar)

    pA_pmbar[2:4, 0:2] += -1.0 * pMinvK_pmbar[:, :]
    pA_pmbar[2:4, 2:4] += -1.0 * pMinvDa_pmbar[:, :]

    # A is not a function of kappa_3 so partial derivative is zero
    pA_pkappa_3 = np.zeros((4, 4))

    return pA_pmbar, pA_pkappa_3

# ----- Mass Matrices

def evalMs():
    Ms = np.zeros((2,2))
    Ms[0,0] = 1.0
    Ms[0,1] = xa_con
    Ms[1,0] = xa_con
    Ms[1,1] = ra_con ** 2
    return Ms

def evalMa(x):
    mbar, kappa_3 = _extractDVs(x)

    Ma = np.zeros((2,2))
    Ma[0,0] = 1.0
    Ma[0,1] = -a_con
    Ma[1,0] = -a_con
    Ma[1,1] = 1.0/8.0 + a_con ** 2
    coeff = 1.0/mbar
    Ma = coeff * Ma
    return Ma

def evalM(x):
    return evalMs() + evalMa(x)

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

def evalMInv(x):
    mbar, kappa_3 = _extractDVs(x)

    tmp = 1.0/8.0 + a_con ** 2
    det = ra_con**2 + tmp/mbar + ra_con**2/mbar + tmp/mbar**2 - xa_con**2 + 2*a_con*xa_con/mbar - (a_con/mbar)**2

    MInv = np.zeros((2,2))
    MInv[0,0] = ra_con** 2 + tmp/mbar
    MInv[0,1] = -(xa_con - a_con/mbar)
    MInv[1,0] = -(xa_con - a_con/mbar)
    MInv[1,1] = 1.0 + 1.0/mbar
    MInv = 1.0 / det * MInv

    return MInv

def evalpMInvpx(x):
    """
    pMInv/px = p(1/c(x) * B(x)) / px = -1/c^2 * pc/px * B + 1/c pB/px
    """
    mbar, kappa_3 = _extractDVs(x)

    tmp = 1.0/8.0 + a_con ** 2
    det = ra_con**2 + tmp/mbar + ra_con**2/mbar + tmp/mbar**2 - xa_con**2 + 2*a_con*xa_con/mbar - (a_con/mbar)**2
    oneOverDet = 1.0 / det

    # mbar only (not a dependence on kappa_3)
    # pc/pmbar
    ddet_dmbar = - tmp/mbar**2 - ra_con**2/mbar**2 - 2*tmp/mbar**3 - 2*a_con*xa_con/mbar**2 + 2*a_con**2/mbar**3
    Minv = evalMInv(x)

    # pB/pmbar
    pB_pmbar = np.zeros((2,2))
    pB_pmbar[0,0] = -tmp/mbar**2
    pB_pmbar[0,1] = -a_con/mbar**2
    pB_pmbar[1,0] = -a_con/mbar**2
    pB_pmbar[1,1] = -1.0/mbar**2

    MInv_pmbar = -oneOverDet * ddet_dmbar * Minv + oneOverDet * pB_pmbar

    MInv_pkappa_3 = np.zeros((2,2))

    return MInv_pmbar, MInv_pkappa_3


# ----- Stiffness Matrices
def evalDa(x):
    mbar, kappa_3 = _extractDVs(x)

    Da = np.zeros((2,2))
    Da[0,0] = 1.0
    Da[0,1] = 1.0 - a_con
    Da[1,0] = -(1.0/2.0 + a_con)
    Da[1,1] = a_con * (a_con - 1.0/2.0)

    coeff = 2*mu_con/mbar
    Da = coeff * Da

    return Da

def evalpDapx(x):
    mbar, kappa_3 = _extractDVs(x)

    # only need mbar (wrt kappa_3 will be zero)
    Da = evalDa(x)
    pDa_pmbar = -1.0/mbar * Da
    return pDa_pmbar


# ----- Stiffness Matrices
def evalKs():
    Ks = np.zeros((2,2))
    Ks[0,0] = Omega_con ** 2
    Ks[1,1] = ra_con ** 2

    return Ks

def evalKa(x):
    mbar, kappa_3 = _extractDVs(x)

    Ka = np.zeros((2,2))
    Ka[0,1] = 1.0
    Ka[1,1] = -(1.0/2.0 + a_con)

    coeff = 2*mu_con**2/mbar
    Ka = coeff * Ka

    return Ka

def evalK(x):
    return evalKs() + evalKa(x)

def evalpKpx(x):
    """
    pK/px = p/px(Ks + Ka) = pKa/px
    """
    return evalpKapx(x)

def evalpKapx(x):
    mbar, kappa_3 = _extractDVs(x)

    # only need mbar (wrt kappa_3 will be zero)
    Ka = evalKa(x)
    pKa_pmbar = -1.0/mbar * Ka
    return pKa_pmbar

# ----------
# Top level
# ----------

def f_LST(v, x):

    """
    Linear stability function.
    """
    # Here we return the real part of the eigenvalue
    return v[-2]

def f_pLST_pv(v, x):
    """
    pLST / pv
    """

    ndof = np.shape(v)[0]
    pLST_pv = np.zeros(ndof)
    pLST_pv[-2] = 1.0

    return pLST_pv
