import numpy as np
import cmath
import copy
import utils as utils
# import matplotlib.pyplot as plt
from plot_utils import *
from scipy import optimize


class Hopf_bif(object):

    '''
        Hopf bifurcation class. 
    '''

    def __init__(self, x, func, func_A, ndof, pfunc_pmu = None, 
                 pfunc_A_pmu = None, pfunc_A_pw = None):

        '''
            Initiate with design var and coeff function.
        '''

        # Design variable
        self.x = x

        # Coeff function
        # dot{w} = func(w, mu, x)
        # A = A(mu, x) = p func / p w
        self.func = func 
        self.func_A = func_A

        self.ndof = ndof

        self.pfunc_pmu = pfunc_pmu

        self.pfunc_A_pmu = pfunc_A_pmu
        self.pfunc_A_pw = pfunc_A_pw

    def solve_steady(self, mu, win=None, tol=1e-4):
        
        ndof = self.ndof
        x = self.x

        # If not initialized, initialize it with 0.
        if (win is None):
            win = np.zeros(ndof)

        # Solve the nonlinear steady state equation
        r = (lambda w: self.func(w, mu, x))
        sol = optimize.newton_krylov(r, win, f_tol = tol)
        # sol = optimize.fsolve(r, win)
        
        return sol

    def solve_eig_R(self, mu, w):

        '''
            Compute the eigenvalue problem (right) with 
            the maximum real part.
        '''

        x = self.x

        # Extract the coeff matrix for the bifurcation point.
        A = self.func_A(w, mu, x)

        eigval_arr, eigvec_arr = np.linalg.eig(A)

        # Get the the maximum real part vector.
        ind = np.argmax(np.real(eigval_arr))

        # We assume the eigenvalue imaginary part to be positive
        eigval = eigval_arr[ind]
        if np.imag(eigval) == 0.0:
            pass
        elif np.imag(eigval) < 0.0:
            ind = np.abs(eigval_arr - np.conj(eigval)).argmin()
            eigval = eigval_arr[ind]

        eigvec = eigvec_arr[:, ind]

        return eigval, eigvec
    
    def solve_eig_L(self, win=None):
        
        eigval_bif_R = self.eigval_bif_R
        eigvec_bif_R = self.eigvec_bif_R
        x = self.x
        w = self.w
        mu = self.mu

        # Extract the coeff matrix for the bifurcation point.
        A = self.func_A(w, mu, x)
        eigval_arr, eigvec_arr = np.linalg.eig(A.T)

        ind = np.argmin(abs(eigval_arr + eigval_bif_R))

        eigval_bif_L = - eigval_bif_R
        eigvec_bif_L = eigvec_arr[:, ind]

        val = np.dot(np.conj(eigvec_bif_R), eigvec_bif_L)
        eigvec_bif_L = eigvec_bif_L / val

        self.eigval_bif_L = eigval_bif_L
        self.eigvec_bif_L = eigvec_bif_L

    def compute_bif_eig_res(self):

        ind = self.ind

        x = self.x
        w = self.w
        mu = self.mu
        eigval_bif_R = self.eigval_bif_R
        eigvec_bif_R = self.eigvec_bif_R

        ndof = self.ndof

        omega = np.imag(eigval_bif_R)

        eigvec_bif_R_r = np.real(eigvec_bif_R)
        eigvec_bif_R_i = np.imag(eigvec_bif_R)

        res = np.zeros(ndof*3+2)

        res[0:ndof] = self.func(w, mu, x)

        

        J = self.func_A(w, mu, x)
        res[ndof:2*ndof] = J.dot(eigvec_bif_R_r) + omega * eigvec_bif_R_i
        res[2*ndof:3*ndof] = J.dot(eigvec_bif_R_i) - omega * eigvec_bif_R_r
        res[3*ndof] = eigvec_bif_R_r.dot(eigvec_bif_R_r) + eigvec_bif_R_i.dot(eigvec_bif_R_i) - 1
        
        ej = np.zeros(ndof)
        ej[ind] = 1.0
        res[3*ndof+1] = np.dot(ej, eigvec_bif_R_i)

        return res
    
    def compute_bif_eig_L_res(self):
    
        # HACK: we leverage the fact that mu, w, and omega are the same with right eigenvalue prob

        ind = self.ind

        x = self.x
        w = self.w
        mu = self.mu
        eigval_bif_R = self.eigval_bif_R
        eigvec_bif_R = self.eigvec_bif_R
        eigval_bif_L = self.eigval_bif_L
        eigvec_bif_L = self.eigvec_bif_L

        ndof = self.ndof

        omega = np.imag(eigval_bif_R)

        eigvec_bif_R_r = np.real(eigvec_bif_R)
        eigvec_bif_R_i = np.imag(eigvec_bif_R)
        eigvec_bif_L_r = np.real(eigvec_bif_L)
        eigvec_bif_L_i = np.imag(eigvec_bif_L)

        res = np.zeros(ndof*3+2)

        res[0:ndof] = self.func(w, mu, x)

        
        J = self.func_A(w, mu, x)
        res[ndof:2*ndof] = J.T.dot(eigvec_bif_L_r) - omega * eigvec_bif_L_i
        res[2*ndof:3*ndof] = J.T.dot(eigvec_bif_L_i) + omega * eigvec_bif_L_r
        res[3*ndof] = eigvec_bif_R_r.dot(eigvec_bif_L_r) + eigvec_bif_R_i.dot(eigvec_bif_L_i) - 1
        res[-1] = - eigvec_bif_R_i.dot(eigvec_bif_L_r) + eigvec_bif_R_r.dot(eigvec_bif_L_i)

        return res

    def solve_bif_eig(self, mu_lower, mu_upper, delta, win=None):

        x = self.x

        if ( mu_upper <= mu_lower ):
            print("Error: Upper bound less or equal to the lower.")
            exit()
        
        # Evaluate function value
        w_lower = self.solve_steady(mu_lower, win=win)
        # print(w_lower)
        eigval_lower, eigvec_lower = self.solve_eig_R(mu_lower, w_lower)

        w_upper = self.solve_steady(mu_upper, win=win)
        # print(w_upper)
        eigval_upper, eigvec_upper = self.solve_eig_R(mu_upper, w_upper)

        # Safety check
        if ( np.real(eigval_lower) * np.real(eigval_upper) > 0 ):
            print("Error: Not sure solution in between.")
            print("np.real(eigval_lower)", np.real(eigval_lower))
            print("np.real(eigval_upper)", np.real(eigval_upper))
            exit()

        # Main loop for the search
        while ( mu_upper - mu_lower > delta ):

            mu_mid = (mu_upper + mu_lower) / 2

            w_mid = self.solve_steady(mu_mid, win=win)
            eigval_mid, eigvec_mid = self.solve_eig_R(mu_mid, w_mid)

            if ( np.real(eigval_mid) == 0 ):
                break

            if ( np.real(eigval_mid) * np.real(eigval_lower) < 0):
                mu_upper = mu_mid
                eigval_upper = eigval_mid
            elif ( np.real(eigval_mid) * np.real(eigval_upper) < 0 ):
                mu_lower = mu_mid
                eigval_lower = eigval_mid
            else:
                print("Error: Something unexpected happened...")

        # Set the solution
        mu = (mu_upper + mu_lower) / 2
        w = self.solve_steady(mu, win=win)

        # Compute the eigenvalue and eigenvector
        eigval, eigvec = self.solve_eig_R(mu, w)

        # Set the eigenvalue to be a pure imaginary number
        eigval = np.imag(eigval) * 1j

        # Rotate the vector 
        self.ind = np.argmax(abs(eigvec))
        theta = np.angle(eigvec[self.ind])
        e_itheta = np.exp(- 1j * theta)
        theta *= e_itheta

        # Set the variables
        self.mu = mu
        self.w = w
        self.eigval_bif_R = eigval
        self.eigvec_bif_R = eigvec

    def set_bif_adjoint_mat(self):

        # Extract design var
        x = self.x

        # Extract state var
        w = self.w
        mu = self.mu
        eigval_bif_R = self.eigval_bif_R
        eigvec_bif_R = self.eigvec_bif_R

        # Cast into real and imag
        omega = np.imag(eigval_bif_R)

        eigvec_bif_R_r = np.real(eigvec_bif_R)
        eigvec_bif_R_i = np.imag(eigvec_bif_R)

        # Dimension of equilibrium point 
        ndof = self.ndof

        # Adjoint matrix assemby
        # Construct the individual component of J^T
        prR_puRT = np.zeros((ndof*3 + 2, ndof*3 + 2))

        pr_pmu = self.pfunc_pmu(w, mu, x)
        pr_pw = self.func_A(w, mu, x)

        pJ_pmu = self.pfunc_A_pmu(w, mu, x)
        pJ_pw = self.pfunc_A_pw(w, mu, x)

        GR_r = np.dot(pJ_pw, eigvec_bif_R_r)
        GR_i = np.dot(pJ_pw, eigvec_bif_R_i)
        
        gR_r = np.dot(pJ_pmu, eigvec_bif_R_r)
        gR_i = np.dot(pJ_pmu, eigvec_bif_R_i)

        ej = np.zeros(ndof)
        ej[self.ind] = 1.0
        I = np.eye(ndof)
        

        # Fill in ...

        # Diagonal
        prR_puRT[0:ndof, 0:ndof] = (pr_pw.T)[:, :]
        prR_puRT[ndof:2*ndof, ndof:2*ndof] = (pr_pw.T)[:, :]
        prR_puRT[2*ndof:3*ndof, 2*ndof:3*ndof] = (pr_pw.T)[:, :]

        # Off-diagonal
        # row 1
        prR_puRT[0:ndof, ndof:2*ndof] = (GR_r.T)[:,:]
        prR_puRT[0:ndof, 2*ndof:3*ndof] = (GR_i.T)[:,:]

        # row 2
        prR_puRT[ndof:2*ndof, 2*ndof:3*ndof] = (- omega * I)[:, :]
        prR_puRT[ndof:2*ndof, 3*ndof] = (2 * eigvec_bif_R_r)[:]

        # row 3
        prR_puRT[2*ndof:3*ndof, ndof:2*ndof] = (omega * I)[:]
        prR_puRT[2*ndof:3*ndof, 3*ndof] = (2 * eigvec_bif_R_i)[:]
        prR_puRT[2*ndof:3*ndof, -1] = ej[:]

        # row 4
        prR_puRT[3*ndof, 0:ndof] = pr_pmu[:]
        prR_puRT[3*ndof, ndof:2*ndof] = gR_r[:]
        prR_puRT[3*ndof, 2*ndof:3*ndof] = gR_i[:]

        # row 5
        prR_puRT[3*ndof + 1, ndof:2*ndof] = eigvec_bif_R_i[:]
        prR_puRT[3*ndof + 1, 2*ndof:3*ndof] = - eigvec_bif_R_r[:]

        self.prR_puRT = prR_puRT

    def set_bif_L_adjoint_mat(self):

        # HACK: we leverage the fact that mu, w, and omega are the same with right eigenvalue prob
        # GR_r, GR_i, gR_r, gR_i can be wrong

        # Extract design var
        x = self.x

        # Extract state var
        w = self.w
        mu = self.mu
        eigval_bif_R = self.eigval_bif_R
        eigvec_bif_R = self.eigvec_bif_R
        eigval_bif_L = self.eigval_bif_L
        eigvec_bif_L = self.eigvec_bif_L

        # Cast into real and imag
        omega = np.imag(eigval_bif_R)

        eigvec_bif_R_r = np.real(eigvec_bif_R)
        eigvec_bif_R_i = np.imag(eigvec_bif_R)
        eigvec_bif_L_r = np.real(eigvec_bif_L)
        eigvec_bif_L_i = np.imag(eigvec_bif_L)

        # Dimension of equilibrium point 
        ndof = self.ndof

        # Adjoint matrix assemby
        # Construct the individual component of J^T
        prL_puLT = np.zeros((ndof*3 + 2, ndof*3 + 2))

        pr_pmu = self.pfunc_pmu(w, mu, x)
        pr_pw = self.func_A(w, mu, x)

        pJ_pmu = self.pfunc_A_pmu(w, mu, x)
        pJ_pw = self.pfunc_A_pw(w, mu, x)

        GL_r = np.dot(pJ_pw.T, eigvec_bif_L_r) 
        GL_i = np.dot(pJ_pw.T, eigvec_bif_L_i) 
        
        gL_r = np.dot(pJ_pmu.T, eigvec_bif_L_r)
        gL_i = np.dot(pJ_pmu.T, eigvec_bif_L_i)
        I = np.eye(ndof)
        

        # Fill in ...

        # Diagonal
        prL_puLT[0:ndof, 0:ndof] = (pr_pw.T)[:, :]
        prL_puLT[ndof:2*ndof, ndof:2*ndof] = (pr_pw)[:, :]
        prL_puLT[2*ndof:3*ndof, 2*ndof:3*ndof] = (pr_pw)[:, :]

        # Off-diagonal
        # row 1
        prL_puLT[0:ndof, ndof:2*ndof] = (GL_r.T)[:,:]
        prL_puLT[0:ndof, 2*ndof:3*ndof] = (GL_i.T)[:,:]

        # row 2
        prL_puLT[ndof:2*ndof, 2*ndof:3*ndof] = (omega * I)[:, :]
        prL_puLT[ndof:2*ndof, -2] = eigvec_bif_R_r[:]
        prL_puLT[ndof:2*ndof, -1] = - eigvec_bif_R_i[:]

        # row 3
        prL_puLT[2*ndof:3*ndof, ndof:2*ndof] = (-omega * I)[:]
        prL_puLT[2*ndof:3*ndof, -2] = eigvec_bif_R_i[:]
        prL_puLT[2*ndof:3*ndof, -1] = eigvec_bif_R_r[:]

        # row 4
        prL_puLT[3*ndof, 0:ndof] = pr_pmu[:]
        prL_puLT[3*ndof, ndof:2*ndof] = gL_r[:]
        prL_puLT[3*ndof, 2*ndof:3*ndof] = gL_i[:]

        # row 5
        prL_puLT[3*ndof + 1, ndof:2*ndof] = - eigvec_bif_L_i[:]
        prL_puLT[3*ndof + 1, 2*ndof:3*ndof] = eigvec_bif_L_r[:]

        self.prL_puLT = prL_puLT

    def solve_bif_adjoint(self, pf_puR):

        # Solve the adjoint equation for the Hopf bifurcation (lower level) 

        # Exract Jacobian matrix
        prR_puRT = self.prR_puRT

        # Solve adjoint
        psi_rR = np.linalg.solve(prR_puRT, pf_puR)

        self.psi_rR = psi_rR

    def solve_bif_L_adjoint(self, pf_puL):
    
        # Solve the adjoint equation for the Hopf bifurcation (lower level) 

        # Exract Jacobian matrix
        prL_puLT = self.prL_puLT

        # Solve adjoint
        psi_rL = np.linalg.solve(prL_puLT, pf_puL)

        self.psi_rL = psi_rL

    def solve_bif_stab_adjoint(self, pf_puL, pf_puR):

        ndof = self.ndof

        eigvec_bif_L = self.eigvec_bif_L

        eigvec_bif_L_r = np.real(eigvec_bif_L)
        eigvec_bif_L_i = np.imag(eigvec_bif_L)

        # pf_puL = pf_pu[0:ndof]
        # pf_puR = pf_pu[ndof:2*ndof]

        # Solve top level adjoint
        self.solve_bif_L_adjoint(pf_puL)

        # Compute the off-diagonal term
        prL_puR = np.zeros((3*ndof+2, 3*ndof+2))
        prL_puR[3*ndof, ndof:2*ndof] = eigvec_bif_L_r[:]
        prL_puR[3*ndof, 2*ndof:3*ndof] = eigvec_bif_L_i[:]
        prL_puR[3*ndof+1, ndof:2*ndof] = eigvec_bif_L_i[:]
        prL_puR[3*ndof+1, 2*ndof:3*ndof] = -eigvec_bif_L_r[:]

        # prL_puR = np.zeros((3*ndof+2, 3*ndof+2))
        # prL_puR = prL_puR.at[3*ndof, ndof:2*ndof].set(eigvec_bif_L_r[:])
        # prL_puR = prL_puR.at[3*ndof, 2*ndof:3*ndof].set(eigvec_bif_L_i[:])
        # prL_puR = prL_puR.at[3*ndof+1, ndof:2*ndof].set(eigvec_bif_L_i[:])
        # prL_puR = prL_puR.at[3*ndof+1, 2*ndof:3*ndof].set(-eigvec_bif_L_r[:])



        # Solve top level adjoint
        self.solve_bif_adjoint(pf_puR - prL_puR.T.dot(self.psi_rL))

    def compute_bif_total_der(self, pf_px, prR_px):

        return pf_px - self.psi_rR.T.dot(prR_px)
    
    def compute_bif_L_total_der(self, pf_px, prL_px):
    
        return pf_px - self.psi_rL.T.dot(prL_px)
    
    def compute_bif_stab_total_der(self, pf_px, prL_px, prR_px):

        return pf_px - self.psi_rL.T.dot(prL_px) - self.psi_rR.T.dot(prR_px)

    def compute_stab(self, func_B, func_C):

        w = self.w
        mu = self.mu
        x = self.x
        ndof = self.ndof

        eigval_bif_R = self.eigval_bif_R

        eigvec_bif_L = self.eigvec_bif_L
        eigvec_bif_R = self.eigvec_bif_R

        # # HACK
        # # coeff = np.exp(1j * 0.327)
        # coeff = 1.0
        # eigvec_bif_L *= coeff
        # eigvec_bif_R *= coeff

        # print("np.vdot(eigvec_bif_L, eigvec_bif_R)", np.vdot(eigvec_bif_L, eigvec_bif_R))

        c1 = func_C(w, mu, x, eigvec_bif_R, eigvec_bif_R, np.conj(eigvec_bif_R))
        h1 = np.dot(np.conj(eigvec_bif_L), c1)

        J = self.func_A(w, mu, x)
        b21 = func_B(w, mu, x, eigvec_bif_R, np.conj(eigvec_bif_R))
        b22 = func_B(w, mu, x, eigvec_bif_R, np.linalg.solve(J, b21))
        h2 = np.dot(np.conj(eigvec_bif_L), b22)

        J3 = 2*eigval_bif_R*np.eye(ndof) - J
        b31 = func_B(w, mu, x, eigvec_bif_R, eigvec_bif_R)
        b32 = func_B(w, mu, x, np.conj(eigvec_bif_R), np.linalg.solve(J3, b31))
        h3 = np.dot(np.conj(eigvec_bif_L), b32)
        

        l = 1.0 / (2.0 * np.imag(eigval_bif_R)) * np.real(h1 - 2 * h2 + h3)

        return l

# if __name__ == "__main__":

#     def func(w, mu, x):
    
#         """
#             Forcing term.
#         """

#         f0 = (mu - x[0]) * w[0] - w[1] + (2.0 * x[0] * x[1] - 1.0) * w[0] ** 3 - 0.1
#         f1 = w[0] + (mu - x[1]) * w[1] + (2.0 * x[1] - 1.0) * w[1] ** 3

#         f = np.zeros(2)
#         f[0] = f0
#         f[1] = f1

#         return f

#     def pfunc_pmu(w, mu, x):

#         """
#             pf / pmu
#         """

#         pfpmu = np.zeros(2)

#         pfpmu[0] = w[0]
#         pfpmu[1] = w[1]

#         return pfpmu

#     def func_A(w, mu, x):
    
#         """
#             pf / pw
#         """

#         # Each entry ...
#         pf0pw0 = (mu - x[0]) + (2.0 * x[0] * x[1] - 1.0) * 3 * w[0] ** 2
#         pf0pw1 = -1.0
#         pf1pw0 = 1.0
#         pf1pw1 = (mu - x[1]) + (2.0 * x[1] - 1.0) * 3 * w[1] ** 2

#         # Fill in
#         pfpw = np.zeros((2, 2))

#         pfpw[0, 0] = pf0pw0
#         pfpw[0, 1] = pf0pw1
#         pfpw[1, 0] = pf1pw0
#         pfpw[1, 1] = pf1pw1

#         return pfpw
    
#     def pfunc_A_pmu(w, mu, x):

#         """
#             p^2f / pwpmu
#         """

#         p2fpwpmu = np.zeros((2, 2))

#         p2fpwpmu[0, 0] = 1.0
#         p2fpwpmu[1, 1] = 1.0

#         return p2fpwpmu

#     def pfunc_A_pw(w, mu, x):

#         """
#             p^2f / pw2
#         """

#         p2fpw2 = np.zeros((2, 2, 2))
#         p2fpw2[0, 0, 0] = (2.0 * x[0] * x[1] - 1.0) * 6 * w[0]
#         p2fpw2[1, 1, 1] = (2.0 * x[1] - 1.0) * 6 * w[1]

#         return p2fpw2

#     def func_B(w, mu, x, q1, q2):

#         f0_B = np.zeros((2, 2), dtype=np.complex128)
#         f0_B[0, 0] = (2.0 * x[0] * x[1] - 1.0) * 6 * w[0]
        
#         f1_B = np.zeros((2, 2), dtype=np.complex128)
#         f1_B[1, 1] = (2.0 * x[1] - 1.0) * 6 * w[1]


#         vec_B = np.zeros(2, dtype=np.complex128)
#         vec_B[0] = q2.T.dot(f0_B.dot(q1))
#         vec_B[1] = q2.T.dot(f1_B.dot(q1))

#         return vec_B
    
#     def func_C(w, mu, x, q1, q2, q3):

#         f0_C = np.zeros((2, 2, 2), dtype=np.complex128)
#         f0_C[0, 0, 0] = (2.0 * x[0] * x[1] - 1.0) * 6

#         f1_C = np.zeros((2, 2, 2), dtype=np.complex128)
#         f1_C[1, 1, 1] = (2.0 * x[1] - 1.0) * 6

#         vec_C = np.zeros(2, dtype=np.complex128)

#         vec_C_1 = np.zeros(2, dtype=np.complex128)
#         vec_C_1[0] = q2.T.dot(f0_C[0, :, :].dot(q1))
#         vec_C_1[1] = q2.T.dot(f0_C[1, :, :].dot(q1))
#         vec_C[0] = vec_C_1.dot(q3)

#         vec_C_2 = np.zeros(2, dtype=np.complex128)
#         vec_C_2[0] = q2.T.dot(f1_C[0, :, :].dot(q1))
#         vec_C_2[1] = q2.T.dot(f1_C[1, :, :].dot(q1))
#         vec_C[1] = vec_C_2.dot(q3)
        
#         return vec_C

#     # ============
#     # Analysis
#     # ============

#     x = [0.2, 0.7]
#     ndof = 2
#     Hopf_bif_obj = Hopf_bif(x, func, func_A, ndof, pfunc_pmu=pfunc_pmu, pfunc_A_pmu=pfunc_A_pmu, pfunc_A_pw=pfunc_A_pw)

#     mu_lower = 0.33
#     mu_upper = 0.8
#     delta = 1e-8
#     Hopf_bif_obj.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

#     Hopf_bif_obj.solve_eig_L()
#     l = Hopf_bif_obj.compute_stab(func_B, func_C)

    

#     # original eigenvalue
#     eigval = Hopf_bif_obj.eigval_bif_R
#     eigvec = Hopf_bif_obj.eigvec_bif_R

#     print("=" * 20)
#     print("Analysis")
#     print("=" * 20)
#     print("mu", Hopf_bif_obj.mu)
#     print("l", l)

#     # ============
#     # Sensitivity
#     # ============

#     # ------------
#     # Right eigenvalue prob
#     # ------------

#     # Finite differences
#     df_dx_FD = np.zeros(len(x))
#     epsilon = 1e-6
#     for i in range(len(x)):

#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj_p = Hopf_bif(x_p, func, func_A, ndof)

#         Hopf_bif_obj_p.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

#         eigval_p = Hopf_bif_obj_p.eigval_bif_R

#         df_dx_FD[i] = (np.imag(eigval_p) - np.imag(eigval)) / epsilon


#     # Adjoint
#     r0 = Hopf_bif_obj.compute_bif_eig_res()
#     pr_px = np.zeros((3*ndof+2, len(x)))
#     for i in range(len(x)):
        
#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj.x = copy.deepcopy(x_p)
#         r_p = Hopf_bif_obj.compute_bif_eig_res()

#         pr_px[:, i] = (r_p[:] - r0[:]) / epsilon
#     Hopf_bif_obj.x = copy.deepcopy(x)

#     pf_px = np.zeros(len(x))

#     pf_pu = np.zeros(3*ndof+2)
#     pf_pu[-1] = 1.0

#     Hopf_bif_obj.set_bif_adjoint_mat()
#     Hopf_bif_obj.solve_bif_adjoint(pf_pu)

#     df_dx_AD = Hopf_bif_obj.compute_bif_total_der(pf_px, pr_px)

#     print("=" * 20)
#     print("Sensitivity")
#     print("=" * 20)

#     print("-"*20)
#     print("Hopf bifurcation (Right)")
#     print("-"*20)


#     print("Finite difference:", df_dx_FD)
#     print("Adjoint:", df_dx_AD)
#     print("-" * 20)

#     # ------------
#     # Left eigenvalue prob
#     # ------------

#     # Hopf_bif_obj.set_bif_L_adjoint_mat(is_lock_qr=True, eigvec_bif_R=eigvec_bif_R)

#     # Finite differences
#     df_dx_FD = np.zeros(len(x))
#     epsilon = 1e-4
#     eigval = eigval = Hopf_bif_obj.eigval_bif_L
#     for i in range(len(x)):

#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj_p = Hopf_bif(x_p, func, func_A, ndof)

#         Hopf_bif_obj_p.solve_bif_eig(mu_lower, mu_upper, delta, win=None)
#         Hopf_bif_obj_p.solve_eig_L()

#         eigval_p = Hopf_bif_obj_p.eigval_bif_L

#         df_dx_FD[i] = (np.imag(eigval_p) - np.imag(eigval)) / epsilon

#     # Adjoint

#     pr_px = np.zeros((3*ndof+2, len(x)))
#     r0 = Hopf_bif_obj.compute_bif_eig_L_res()
#     for i in range(len(x)):
        
#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj.x = copy.deepcopy(x_p)
#         r_p = Hopf_bif_obj.compute_bif_eig_L_res()

#         pr_px[:, i] = (r_p[:] - r0[:]) / epsilon
#     Hopf_bif_obj.x = copy.deepcopy(x)

#     pf_px = np.zeros(len(x))

#     pf_pu = np.zeros(3*ndof+2)
#     pf_pu[-1] = 1.0

#     Hopf_bif_obj.set_bif_L_adjoint_mat()
#     Hopf_bif_obj.solve_bif_L_adjoint(pf_pu)

#     df_dx_AD = Hopf_bif_obj.compute_bif_L_total_der(pf_px, pr_px)


#     print("-"*20)
#     print("Hopf bifurcation (Left)")
#     print("-"*20)


#     print("Finite difference:", df_dx_FD)
#     print("Adjoint:", df_dx_AD)
#     print("-" * 20)


#     # ------------
#     # Stability eigenvalue prob
#     # ------------

#     # Finite differences
#     dl_dx_FD = np.zeros(len(x))
#     epsilon = 1e-6
#     for i in range(len(x)):

#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj_p = Hopf_bif(x_p, func, func_A, ndof)

#         Hopf_bif_obj_p.solve_bif_eig(mu_lower, mu_upper, delta, win=None)

#         Hopf_bif_obj_p.solve_eig_L()

#         l_p = Hopf_bif_obj_p.compute_stab(func_B, func_C)



#         dl_dx_FD[i] = (l_p - l) / epsilon


#     # Adjoint

#     # pr / px 
#     prL_px = np.zeros((3*ndof+2, len(x)))
#     prR_px = np.zeros((3*ndof+2, len(x)))

#     rL_0 = Hopf_bif_obj.compute_bif_eig_L_res()
#     rR_0 = Hopf_bif_obj.compute_bif_eig_res()

#     for i in range(len(x)):
        
#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj.x = copy.deepcopy(x_p)
#         rL_p = Hopf_bif_obj.compute_bif_eig_L_res()
#         rR_p = Hopf_bif_obj.compute_bif_eig_res()

#         prL_px[:, i] = (rL_p[:] - rL_0[:]) / epsilon
#         prR_px[:, i] = (rR_p[:] - rR_0[:]) / epsilon

#     Hopf_bif_obj.x = copy.deepcopy(x)

#     # pf / px

#     pl_px = np.zeros(len(x))

#     for i in range(len(x)):

#         x_p = copy.deepcopy(x)
#         x_p[i] += epsilon

#         Hopf_bif_obj.x = copy.deepcopy(x_p)

#         l_p = Hopf_bif_obj.compute_stab(func_B, func_C)
#         pl_px[i] = (l_p - l)/epsilon

#     Hopf_bif_obj.x = copy.deepcopy(x)

#     # pf / pu
#     pl_puL = np.zeros(3*ndof+2)
#     pl_puR = np.zeros(3*ndof+2)

#     w_0 = copy.deepcopy(Hopf_bif_obj.w)
#     eigvec_bif_R_0 = copy.deepcopy(Hopf_bif_obj.eigvec_bif_R)
#     eigvec_bif_L_0 = copy.deepcopy(Hopf_bif_obj.eigvec_bif_L)
#     mu_0 = copy.deepcopy(Hopf_bif_obj.mu)
#     eigval_bif_R_0 = copy.deepcopy(Hopf_bif_obj.eigval_bif_R)

#     for i in range(3*ndof+2):

#         if (i < ndof):
#             w_p = copy.deepcopy(Hopf_bif_obj.w)
#             w_p[i] += epsilon

#             Hopf_bif_obj.w = copy.deepcopy(w_p)
#         elif ((i>=ndof) and (i<2*ndof)):
#             eigvec_bif_R_p = copy.deepcopy(Hopf_bif_obj.eigvec_bif_R)
#             eigvec_bif_R_p[i - ndof] += epsilon

#             Hopf_bif_obj.eigvec_bif_R = copy.deepcopy(eigvec_bif_R_p)
#         elif ((i>=2*ndof) and (i<3*ndof)):
#             eigvec_bif_R_p = copy.deepcopy(Hopf_bif_obj.eigvec_bif_R)
#             eigvec_bif_R_p[i - 2*ndof] += 1j*epsilon

#             Hopf_bif_obj.eigvec_bif_R = copy.deepcopy(eigvec_bif_R_p)
#         elif (i==3*ndof):
#             mu_p = copy.deepcopy(Hopf_bif_obj.mu)
#             mu_p += epsilon

#             Hopf_bif_obj.mu = copy.deepcopy(mu_p)
#         elif (i==3*ndof + 1):
#             eigval_bif_R_p = copy.deepcopy(Hopf_bif_obj.eigval_bif_R)
#             eigval_bif_R_p += 1j*epsilon

#             Hopf_bif_obj.eigval_bif_R = copy.deepcopy(eigval_bif_R_p)
        
#         l_p = Hopf_bif_obj.compute_stab(func_B, func_C)
#         pl_puR[i] = (l_p - l)/epsilon

#         if (i < ndof):
#             Hopf_bif_obj.w = copy.deepcopy(w_0)
#         elif ((i>=ndof) and (i<2*ndof)):
#             Hopf_bif_obj.eigvec_bif_R = copy.deepcopy(eigvec_bif_R_0)
#         elif ((i>=2*ndof) and (i<3*ndof)):
#             Hopf_bif_obj.eigvec_bif_R = copy.deepcopy(eigvec_bif_R_0)
#         elif (i==3*ndof):
#             Hopf_bif_obj.mu = copy.deepcopy(mu_0)
#         elif (i==3*ndof + 1):
#             Hopf_bif_obj.eigval_bif_R = copy.deepcopy(eigval_bif_R_0)

#     for i in range(3*ndof+2):
    
#         if ((i>=ndof) and (i<2*ndof)):
#             eigvec_bif_L_p = copy.deepcopy(Hopf_bif_obj.eigvec_bif_L)
#             eigvec_bif_L_p[i - ndof] += epsilon

#             Hopf_bif_obj.eigvec_bif_L = copy.deepcopy(eigvec_bif_L_p)
#         elif ((i>=2*ndof) and (i<3*ndof)):
#             eigvec_bif_L_p = copy.deepcopy(Hopf_bif_obj.eigvec_bif_L)
#             eigvec_bif_L_p[i - 2*ndof] += 1j*epsilon

#             Hopf_bif_obj.eigvec_bif_L = copy.deepcopy(eigvec_bif_L_p)
        
#         l_p = Hopf_bif_obj.compute_stab(func_B, func_C)
#         pl_puL[i] = (l_p - l)/epsilon

#         if ((i>=ndof) and (i<2*ndof)):
#             Hopf_bif_obj.eigvec_bif_L = copy.deepcopy(eigvec_bif_L_0)
#         elif ((i>=2*ndof) and (i<3*ndof)):
#             Hopf_bif_obj.eigvec_bif_L = copy.deepcopy(eigvec_bif_L_0)

#     Hopf_bif_obj.solve_bif_stab_adjoint(pl_puL, pl_puR)
#     dl_dx_AD = Hopf_bif_obj.compute_bif_stab_total_der(pl_px, prL_px, prR_px)

#     # Alternative lyp AD implementation of adjoint
#     import jax.numpy as np
#     import jax
#     from jax import jit, jacfwd, jacrev, vmap, vjp
#     import jax.lax as lax
#     from jax import random, jacfwd

#     jax.config.update('jax_enable_x64', True)
#     # jax.config.update("jax_traceback_filtering", "off")

#     def func_A(w, mu, x):

#         # Each entry ...
#         pf0pw0 = (mu - x[0]) + (2.0 * x[0] * x[1] - 1.0) * 3 * w[0] ** 2
#         pf0pw1 = -1.0
#         pf1pw0 = 1.0
#         pf1pw1 = (mu - x[1]) + (2.0 * x[1] - 1.0) * 3 * w[1] ** 2

#         # Fill in
#         pfpw = np.zeros((2, 2), dtype=np.complex128)
#         pfpw = pfpw.at[0, 0].set(pf0pw0)
#         pfpw = pfpw.at[0, 1].set(pf0pw1)
#         pfpw = pfpw.at[1, 0].set(pf1pw0)
#         pfpw = pfpw.at[1, 1].set(pf1pw1)

#         return pfpw
    
#     def func_B(w, mu, x, q1, q2):
#         f0_B = np.zeros((2, 2), dtype=np.complex128)
#         f0_B = f0_B.at[0, 0].set((2.0 * x[0] * x[1] - 1.0) * 6 * w[0])

#         f1_B = np.zeros((2, 2), dtype=np.complex128)
#         f1_B = f1_B.at[1, 1].set((2.0 * x[1] - 1.0) * 6 * w[1])

#         vec_B = np.zeros(2, dtype=np.complex128)
#         vec_B = vec_B.at[0].set(np.dot(q2, np.dot(f0_B, q1)))
#         vec_B = vec_B.at[1].set(np.dot(q2, np.dot(f1_B, q1)))

#         return vec_B
    
#     def func_C(w, mu, x, q1, q2, q3):
#         f0_C = np.zeros((2, 2, 2), dtype=np.complex128)
#         f0_C = f0_C.at[0, 0, 0].set((2.0 * x[0] * x[1] - 1.0) * 6)

#         f1_C = np.zeros((2, 2, 2), dtype=np.complex128)
#         f1_C = f1_C.at[1, 1, 1].set((2.0 * x[1] - 1.0) * 6)

#         vec_C = np.zeros(2, dtype=np.complex128)

#         vec_C_1 = np.zeros(2, dtype=np.complex128)
#         vec_C_1 = vec_C_1.at[0].set(q2.dot(f0_C[0].dot(q1)))
#         vec_C_1 = vec_C_1.at[1].set(q2.dot(f0_C[1].dot(q1)))
#         vec_C = vec_C.at[0].set(vec_C_1.dot(q3))

#         vec_C_2 = np.zeros(2, dtype=np.complex128)
#         vec_C_2 = vec_C_2.at[0].set(q2.dot(f1_C[0].dot(q1)))
#         vec_C_2 = vec_C_2.at[1].set(q2.dot(f1_C[1].dot(q1)))
#         vec_C = vec_C.at[1].set(vec_C_2.dot(q3))

#         return vec_C

#     w_0 = copy.deepcopy(Hopf_bif_obj.w)
#     eigvec_bif_R_0 = copy.deepcopy(Hopf_bif_obj.eigvec_bif_R)
#     eigvec_bif_L_0 = copy.deepcopy(Hopf_bif_obj.eigvec_bif_L)
#     mu_0 = copy.deepcopy(Hopf_bif_obj.mu)
#     eigval_bif_R_0 = copy.deepcopy(Hopf_bif_obj.eigval_bif_R)
#     x_0 = np.asarray(x)
#     print("x_0", x_0)

#     omega_0 = np.imag(eigval_bif_R_0)

#     lyp_obj = lyp.Lyapunov(func_A, func_B, func_C, omega_0, eigvec_bif_L_0, eigvec_bif_R_0, x_0, w_0, mu_0)


#     pl_px_AD = lyp_obj.compute_plpx()

#     pl_puL_AD = lyp_obj.compute_plpuL()
#     pl_puR_AD = lyp_obj.compute_plpuR()

#     print("pl_px_AD", pl_px_AD, "pl_px", pl_px, "diff", pl_px_AD - pl_px)
#     print("pl_puL_AD", pl_puL_AD, "pl_puL", pl_puL, "diff", pl_puL_AD - pl_puL)
#     print("pl_puR_AD", pl_puR_AD, "pl_puR", pl_puR, "diff", pl_puR_AD - pl_puR)

#     import numpy as np

#     # Convert to regular np array
#     pl_px_AD = np.array(pl_px_AD)
#     pl_puL_AD = np.array(pl_puL_AD)
#     pl_puR_AD = np.array(pl_puR_AD)

#     Hopf_bif_obj.solve_bif_stab_adjoint(pl_puL_AD, pl_puR_AD)
#     dl_dx_AD_AD = Hopf_bif_obj.compute_bif_stab_total_der(pl_px_AD, prL_px, prR_px)

#     print("-"*20)
#     print("Hopf bifurcation stability sens")
#     print("-"*20)


#     print("Finite difference:", dl_dx_FD)
#     print("Adjoint:", dl_dx_AD)
#     print("Adjoint+lyp AD:", dl_dx_AD_AD)
#     print("-" * 20)

