# import autograd.numpy as np
import numpy as np


class ae_forbif():
    def __init__(self, kappa_5, Omega, r_alpha, x_alpha, a):
        self.kappa_5 = kappa_5
        self.Omega = Omega
        self.r_alpha = r_alpha
        self.x_alpha = x_alpha
        self.a = a

    def func(self, w, mu, x):
        
        """
            Forcing term.
        """

        # Constants
        a = self.a
        Omega = self.Omega
        r_alpha = self.r_alpha
        x_alpha = self.x_alpha
        kappa_5 = self.kappa_5
        theta = 0 # (2 / 180.0) * np.pi

        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]

        # Extract state var
        alpha = w[1]


        Ms = np.zeros((2, 2))
        Ms[0, 0] = 1.0
        Ms[0, 1] = x_alpha
        Ms[1, 0] = x_alpha
        Ms[1, 1] = r_alpha**2

        Ma = np.zeros((2, 2))
        Ma[0, 0] = 1.0
        Ma[0, 1] = - a
        Ma[1, 0] = - a
        Ma[1, 1] = (0.125 + a**2)
        Ma *= (1.0 / mbar)

        M = Ms + Ma
        Minv = np.linalg.inv(M)
        Minv12 = Minv[0, 1]
        Minv22 = Minv[1, 1]

        Ks = np.zeros((2, 2))
        Ks[0, 0] = Omega**2
        Ks[1, 1] = r_alpha**2

        Ka = np.zeros((2, 2))
        Ka[0, 1] = 1.0
        Ka[1, 1] = - 0.5 - a
        Ka *= (2 * mu**2) / mbar
        
        K = Ks + Ka

        Da = np.zeros((2, 2))
        Da[0, 0] = 1.0
        Da[0, 1] = 1.0 - a
        Da[1, 0] = - 0.5 - a
        Da[1, 1] = a * (a - 0.5)
        Da *= (2.0 * mu) / mbar

        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)[:, :]
        A[2:4, 0:2] = - Minv.dot(K)
        A[2:4, 2:4] = - Minv.dot(Da)

        # Nonlinear force
        Fnl = np.zeros(4)
        Fnl[2] = - Minv12 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)
        Fnl[3] = - Minv22 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)

        # Total force
        f = np.zeros(4)
        f[:] += A.dot(w) + Fnl[:]

        return f

    def func_A(self, w, mu, x):

        """
            pf / pw
        """

        # Constants
        a = self.a
        Omega = self.Omega
        r_alpha = self.r_alpha
        x_alpha = self.x_alpha
        kappa_5 = self.kappa_5
        theta = 0 # (2 / 180.0) * np.pi

        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]

        # Extract state var
        alpha = w[1]


        Ms = np.zeros((2, 2))
        Ms[0, 0] = 1.0
        Ms[0, 1] = x_alpha
        Ms[1, 0] = x_alpha
        Ms[1, 1] = r_alpha**2

        Ma = np.zeros((2, 2))
        Ma[0, 0] = 1.0
        Ma[0, 1] = - a
        Ma[1, 0] = - a
        Ma[1, 1] = (0.125 + a**2)
        Ma *= (1.0 / mbar)

        M = Ms + Ma
        Minv = np.linalg.inv(M)
        Minv12 = Minv[0, 1]
        Minv22 = Minv[1, 1]

        Ks = np.zeros((2, 2))
        Ks[0, 0] = Omega**2
        Ks[1, 1] = r_alpha**2

        Ka = np.zeros((2, 2))
        Ka[0, 1] = 1.0
        Ka[1, 1] = - 0.5 - a
        Ka *= (2 * mu**2) / mbar
        
        K = Ks + Ka

        Da = np.zeros((2, 2))
        Da[0, 0] = 1.0
        Da[0, 1] = 1.0 - a
        Da[1, 0] = - 0.5 - a
        Da[1, 1] = a * (a - 0.5)
        Da *= (2.0 * mu) / mbar

        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)[:, :]
        A[2:4, 0:2] = - Minv.dot(K)
        A[2:4, 2:4] = - Minv.dot(Da)

        # Nonlinear force
        pFnlpw = np.zeros((4, 4))

        Fnl = np.zeros(4)
        Fnl[2] = - Minv12 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)
        Fnl[3] = - Minv22 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)

        pFnlpw[2, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)
        pFnlpw[3, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)

        # Total force
        pfpw = np.zeros((4, 4))
        pfpw[:, :] += A[:, :]
        pfpw[:, :] += pFnlpw[:, :]

        return pfpw

    def func_B(self, w, mu, x, q1, q2):
        
        """
            p2f / pw2
        """

        # Constants
        a = self.a
        Omega = self.Omega
        r_alpha = self.r_alpha
        x_alpha = self.x_alpha
        kappa_5 = self.kappa_5
        theta = 0 # (2 / 180.0) * np.pi

        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]

        # Extract state var
        alpha = w[1]

        Ms = np.zeros((2, 2))
        Ms[0, 0] = 1.0
        Ms[0, 1] = x_alpha
        Ms[1, 0] = x_alpha
        Ms[1, 1] = r_alpha**2

        Ma = np.zeros((2, 2))
        Ma[0, 0] = 1.0
        Ma[0, 1] = - a
        Ma[1, 0] = - a
        Ma[1, 1] = (0.125 + a**2)
        Ma *= (1.0 / mbar)

        M = Ms + Ma
        Minv = np.linalg.inv(M)
        Minv12 = Minv[0, 1]
        Minv22 = Minv[1, 1]

        # Nonlinear force
        pFnlpw = np.zeros((4, 4))

        Fnl = np.zeros(4)
        Fnl[2] = - Minv12 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)
        Fnl[3] = - Minv22 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)

        pFnlpw[2, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)
        pFnlpw[3, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)


        # Total force
        p2fpw2= np.zeros((4, 4, 4))
        p2fpw2[2, 1, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 6 * (alpha + theta) + 20 * kappa_5 * (alpha + theta)**3)
        p2fpw2[3, 1, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 6 * (alpha + theta) + 20 * kappa_5 * (alpha + theta)**3)

        # print("p2fpw2", p2fpw2)

        vec_B = np.zeros(4, dtype=complex)
        for i in range(4):

            vec_B[i] = np.dot(p2fpw2[i, :, :].dot(q1), q2)
            # print("p2fpw2[i, :, :]", p2fpw2[i, :, :])
            # print("q1", q1)
            # print("q2", q2)
            # print("p2fpw2[i, :, :].dot(q1)", p2fpw2[i, :, :].dot(q1))
            # print("vec_B", vec_B)
            
        return vec_B

    def func_C(self, w, mu, x, q1, q2, q3):

        """
            p3f / pw3
        """

        # Constants
        a = self.a
        Omega = self.Omega
        r_alpha = self.r_alpha
        x_alpha = self.x_alpha
        kappa_5 = self.kappa_5
        theta = 0 # (2 / 180.0) * np.pi

        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]

        # Extract state var
        alpha = w[1]

        Ms = np.zeros((2, 2))
        Ms[0, 0] = 1.0
        Ms[0, 1] = x_alpha
        Ms[1, 0] = x_alpha
        Ms[1, 1] = r_alpha**2

        Ma = np.zeros((2, 2))
        Ma[0, 0] = 1.0
        Ma[0, 1] = - a
        Ma[1, 0] = - a
        Ma[1, 1] = (0.125 + a**2)
        Ma *= (1.0 / mbar)

        M = Ms + Ma
        Minv = np.linalg.inv(M)
        Minv12 = Minv[0, 1]
        Minv22 = Minv[1, 1]

        # Nonlinear force
        pFnlpw = np.zeros((4, 4))

        Fnl = np.zeros(4)
        Fnl[2] = - Minv12 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)
        Fnl[3] = - Minv22 * r_alpha**2 * (kappa_3 * (alpha + theta)**3 + kappa_5 * (alpha + theta)**5)

        pFnlpw[2, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)
        pFnlpw[3, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 3 * (alpha + theta)**2 + 5 * kappa_5 * (alpha + theta)**4)


        # Total force
        p3fpw3= np.zeros((4, 4, 4, 4))
        p3fpw3[2, 1, 1, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 6 + 60 * kappa_5 * (alpha + theta)**2)
        p3fpw3[3, 1, 1, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 6 + 60 * kappa_5 * (alpha + theta)**2)

        vec_C = np.zeros(4, dtype=complex)
        for i in range(4):
            vec_inter = np.zeros(4, dtype=complex)
            for j in range(4):
                vec_inter[j] = np.dot(p3fpw3[i, j, :, :].dot(q1), q2)
            
            vec_C[i] = np.dot(vec_inter, q3)

        return vec_C


    def pfunc_pmu(self, w, mu, x):

        """
            pf / pmu
        """

        epsilon = 1e-6

        func0 = self.func(w, mu, x)
        funcp = self.func(w, mu+epsilon, x)
        
        return (funcp - func0) / epsilon

    def pfunc_A_pmu(self, w, mu, x):

        """
            pA / pmu
        """

        epsilon = 1e-6

        A0 = self.func_A(w, mu, x)
        Ap = self.func_A(w, mu+epsilon, x)
        
        return (Ap - A0) / epsilon

    def pfunc_A_pw(self, w, mu, x):

        """
            p2f / pw2
        """

        # Constants
        a = self.a
        Omega = self.Omega
        r_alpha = self.r_alpha
        x_alpha = self.x_alpha
        kappa_5 = self.kappa_5
        theta = 0 # (2 / 180.0) * np.pi

        # Extract design var
        mbar = x[0]
        kappa_3 = x[1]

        # Extract state var
        alpha = w[1]

        Ms = np.zeros((2, 2))
        Ms[0, 0] = 1.0
        Ms[0, 1] = x_alpha
        Ms[1, 0] = x_alpha
        Ms[1, 1] = r_alpha**2

        Ma = np.zeros((2, 2))
        Ma[0, 0] = 1.0
        Ma[0, 1] = - a
        Ma[1, 0] = - a
        Ma[1, 1] = (0.125 + a**2)
        Ma *= (1.0 / mbar)

        M = Ms + Ma
        Minv = np.linalg.inv(M)
        Minv12 = Minv[0, 1]
        Minv22 = Minv[1, 1]

        # Total force
        p2fpw2= np.zeros((4, 4, 4))
        p2fpw2[2, 1, 1] = - Minv12 * r_alpha**2 * (kappa_3 * 6 * (alpha + theta) + 20 * kappa_5 * (alpha + theta)**3)
        p2fpw2[3, 1, 1] = - Minv22 * r_alpha**2 * (kappa_3 * 6 * (alpha + theta) + 20 * kappa_5 * (alpha + theta)**3)
    
        return p2fpw2

    def force(self, mu, w, x):
        
        return self.func(w, mu, x)

    def pforcepw(self, mu, w, x):

        return self.func_A(w, mu, x)