import numpy as np
from scipy import optimize
import copy

class lst(object):

    """
        LST (linear stability) analysis and optimization code.
        The underlying dynamics:
            dot{w} + R(w, x) = 0
    """

    def __init__(self, ndof, x, f_res, f_int_dict, f_pres_pw, f_pres_px, f_pint_pw_dict, f_pint_px_dict, f_p2res_pw2_T, f_p2res_pwpx_T, f_int_top_dict, f_pint_top_pv_dict, f_pint_top_px_dict = None, f_pint_top_pJacobian_dict = None, useAnalyticHessian = True):

        # Degree of freedom
        self.ndof = ndof

        # Design variable
        self.x = x
        self.nx = len(x)

        # Nonlinear objective
        self.nonlinear_eqn_obj = nonlinear_eqn(ndof, x, f_res, f_int_dict, f_pres_pw, f_pres_px, f_pint_pw_dict, f_pint_px_dict)

        # Eigenvalue objective
        self.eig_eqn_obj = eig_eqn(ndof, x, None, f_pres_pw, f_int_top_dict, f_pint_top_pv_dict, f_pint_px_dict = f_pint_top_px_dict, f_pint_pJacobian_dict = f_pint_top_pJacobian_dict)

        #
        self.f_p2res_pw2_T = f_p2res_pw2_T
        self.f_p2res_pwpx_T = f_p2res_pwpx_T
        self.f_pres_pw = f_pres_pw
        self.f_pres_px = f_pres_px

        # Dictionary for function of interest: LST and others
        self.f_int_val_dict = {}
        self.dint_dx_dict = {}

        # Whether use analytic hessian or FD
        self.useAnalyticHessian = useAnalyticHessian

    def set_x(self, x):

        self.x = x

        # Set both the bottom nonlinear equation and top eigenvalue equation design vars
        self.nonlinear_eqn_obj.set_x(self.x)
        self.eig_eqn_obj.set_x(self.x)

    def get_w(self):

        # Extract bottom nonlinear equation state vars

        return self.w

    def get_v(self):

        # Extract top eigenvalue equation state vars

        return self.v

    def solve(self, win = None):

        # Solve the underlying nonlinear equation
        self.nonlinear_eqn_obj.solve(win = win)
        self.w = self.nonlinear_eqn_obj.get_w()

        # Solve the eigenvalue problem
        self.eig_eqn_obj.set_w(self.w)
        self.eig_eqn_obj.solve()

        self.v = self.eig_eqn_obj.get_v()

    def compute_int(self, key):

        '''
            Compute function of interest.
            It could be either user specified or the LST measure.
        '''

        if ((key == "LST") or (key == "loss")):
            # LST measure.
            f_int_val = self.eig_eqn_obj.compute_int(key)

        else:
            # Other functions of interest.
            f_int_val = self.nonlinear_eqn_obj.compute_int(key)

        self.f_int_val_dict[key] = f_int_val

        return f_int_val

    def solve_adjoint_top(self, key):

        self.eig_eqn_obj.solve_adjoint(key)
        self.eig_eqn_obj.compute_total_der(key)

    def get_jac(self):

        return self.f_pres_pw(self.w, self.x)

    def set_wbar(self, key):

        w = self.w
        x = self.x

        f_p2res_pw2_T = self.f_p2res_pw2_T
        f_pres_pw = self.f_pres_pw

        [phi1, qr, phi2, qi] = self.eig_eqn_obj.get_total_der(key)

        # HACK
        if self.useAnalyticHessian:
            pres_pw_bar = - np.outer(phi1, qr) - np.outer(phi2, qi)
            self.wbar = f_p2res_pw2_T(w, x, pres_pw_bar)
        else:
            h = 1e-4
            comp1 =  - (f_pres_pw(w + h * qr, x).T.dot(phi1) - f_pres_pw(w - h * qr, x).T.dot(phi1)) / (2 * h)
            comp2 =  - (f_pres_pw(w + h * qi, x).T.dot(phi2) - f_pres_pw(w - h * qi, x).T.dot(phi2)) / (2 * h)
            self.wbar = comp1 + comp2

    def set_xbar(self, key):

        w = self.w
        x = self.x

        f_p2res_pwpx_T = self.f_p2res_pwpx_T
        f_pres_px = self.f_pres_px

        [phi1, qr, phi2, qi] = self.eig_eqn_obj.get_total_der(key)

        # HACK
        if self.useAnalyticHessian:
            pres_pw_bar = - np.outer(phi1, qr) - np.outer(phi2, qi)
            self.xbar = f_p2res_pwpx_T(w, x, pres_pw_bar)
        else:
            h = 1e-4
            comp1 = - (f_pres_px(w + h * qr, x).T.dot(phi1) - f_pres_px(w - h * qr, x).T.dot(phi1)) / (2 * h)
            comp2 = - (f_pres_px(w + h * qi, x).T.dot(phi2) - f_pres_px(w - h * qi, x).T.dot(phi2)) / (2 * h)
            self.xbar = comp1 + comp2

    def f_pLST_pw(self, x, w):

        return self.wbar

    def f_pLST_px(self, x, w):

        nx = self.nx

        return np.zeros(nx)

    def compute_total_der(self, key):

        '''
            Compute the total derivative.
        '''

        if ((key == "LST") or (key == "loss")):

            self.solve_adjoint_top(key)

            # Construct the seed
            self.set_wbar(key)
            self.set_xbar(key)

            # Construct the function to be used in the lower level
            self.nonlinear_eqn_obj.add_to_f_pint_pw_dict(self.f_pLST_pw, key)
            self.nonlinear_eqn_obj.add_to_f_pint_px_dict(self.f_pLST_px, key)

            # Solve the lower level adjoint and the total derivative
            self.nonlinear_eqn_obj.set_adjoint_RHS(key)
            self.nonlinear_eqn_obj.solve_adjoint(key)
            self.nonlinear_eqn_obj.compute_total_der(key)

            # Get the derivative
            dint_dx = self.nonlinear_eqn_obj.get_total_der(key)

            dint_dx += self.xbar

        else:
            # Other functions of interest.
            self.nonlinear_eqn_obj.solve_adjoint(key)
            self.nonlinear_eqn_obj.compute_total_der(key)

            dint_dx = self.nonlinear_eqn_obj.get_total_der(key)

        self.dint_dx_dict[key] = dint_dx

    def get_total_der(self, key):

        return self.dint_dx_dict[key]


class nonlinear_eqn(object):

    """
        Bottom level nonlinear equation.
            R(w, x) = 0.
    """

    def __init__(self, ndof, x, f_res, f_int_dict, f_pres_pw, f_pres_px, f_pint_pw_dict, f_pint_px_dict):

        '''
            Initialize the variables.
        '''

        # Degree of freedom
        self.ndof = ndof

        # Design variable
        self.x = x

        # Residual function
        self.f_res = f_res

        # Function of interest
        self.f_int_dict = f_int_dict
        self.f_int_val_dict = {}

        # Adjoint
        # p R / p w
        self.f_pres_pw = f_pres_pw

        # p R / p x
        self.f_pres_px = f_pres_px

        # p f / p w
        self.f_pint_pw_dict = f_pint_pw_dict
        self.pint_pw_dict = {}

        # p f / p x
        self.f_pint_px_dict = f_pint_px_dict
        self.pint_px_dict = {}

        self.psi_dict = {}
        self.dint_dx_dict = {}

    def set_x(self, x):

        self.x = x

    def solve(self, win = None, tol = 1e-10):

        '''
            Solve the nonlinear equation
                R(w, x) = 0.
        '''

        ndof = self.ndof
        x = self.x

        # If not initialized, initialize it with 0.
        if (win is None):
            win = np.zeros(ndof)

        # Solve the nonlinear steady state equation
        r = (lambda w: self.f_res(w, x))
        sol = optimize.newton_krylov(r, win, f_tol = tol)

        self.w = sol

        return self.w

    def add_to_f_pint_pw_dict(self, f_pint_pw, key):

        self.f_pint_pw_dict[key] = f_pint_pw

    def add_to_f_pint_px_dict(self, f_pint_px, key):

        self.f_pint_px_dict[key] = f_pint_px

    def get_w(self):

        return self.w

    def compute_int(self, key):

        '''
            Compute the function of interest.
        '''

        f_int = self.f_int_dict[key]
        val = f_int(self.w, self.x)

        self.f_int_val_dict[key] = val

        return val

    def set_adjoint_coeff(self):

        '''
            Set the adjoint equation coeff matrix.
        '''

        # Extract variables
        ndof = self.ndof
        w = self.w
        x = self.x

        # Get the matrix
        pres_pw_T = self.f_pres_pw(w, x)

        self.pres_pw_T = pres_pw_T.T

    def set_adjoint_RHS(self, key):

        '''
            Set the RHS of the adjoint equation.
        '''

        f_pint_pw = self.f_pint_pw_dict[key]
        pint_pw = f_pint_pw(self.w, self.x)
        self.pint_pw_dict[key] = pint_pw

    def solve_adjoint(self, key):

        '''
            Solve the adjoint equation.
        '''

        self.set_adjoint_coeff()
        self.set_adjoint_RHS(key)

        pres_pw_T = self.pres_pw_T
        pint_pw = self.pint_pw_dict[key]

        psi = np.linalg.solve(pres_pw_T, pint_pw)

        self.psi_dict[key] = psi

    def compute_total_der(self, key):

        '''
            Compute the total derivative.
        '''

        # Extract variables
        ndof = self.ndof
        w = self.w
        x = self.x
        psi_dict = self.psi_dict

        f_pres_px = self.f_pres_px
        f_pint_px = self.f_pint_px_dict[key]

        pres_px = f_pres_px(w, x)
        pint_px = f_pint_px(w, x)
        psi = psi_dict[key]

        dint_dx = pint_px - pres_px.T.dot(psi)
        self.dint_dx_dict[key] = dint_dx

    def get_total_der(self, key):

        '''
            Extract the total derivative.
        '''

        return self.dint_dx_dict[key]

class eig_eqn(object):

    """
        Eigenvalue derivative object.
    """

    def __init__(self, ndof, x, w, f_Jacobian, f_int_dict, f_pint_pv_dict, f_pint_px_dict = None, f_pint_pJacobian_dict = None):

        # State variable size
        self.ndof = ndof

        # Design variable
        self.x = x
        self.nx = len(x)

        # Steady-state state variable
        self.w = w

        # Eigenvalue state variable
        self.v = np.zeros(2 * ndof + 2)

        # p res / p w
        self.f_Jacobian = f_Jacobian

        # Function of interest, int
        self.f_int_dict = f_int_dict

        # p int / p v
        self.f_pint_pv_dict = f_pint_pv_dict

        # p int / p x
        self.f_pint_px_dict = f_pint_px_dict

        # p int / p x
        self.f_pint_pJacobian_dict = f_pint_pJacobian_dict

        #
        self.f_int_val_dict = {}
        self.pint_pv_dict = {}
        self.phi_dict = {}
        self.dint_dJacobian_dict = {}

    def set_x(self, x):

        self.x = x

    def set_w(self, w):

        # Set the ``background'' state var from the bottom nonlinear equation solution.

        self.w = w

    def get_v(self):

        return self.v

    def solve(self):

        '''
            Solve the top level equation;
                Rhat(v, J) = 0.
        '''

        # Extract the variable
        w = self.w
        x = self.x
        ndof = self.ndof

        # Set the Jacobian
        self.Jacobian = self.f_Jacobian(w, x)

        # Solve the eigenvalues and eigenvectors
        eigval_all, eigvec_all = np.linalg.eig(self.Jacobian)

        # Extract the eigenpair such that the eigenvalue with the largest real part.
        ind = np.argmax(np.real(eigval_all))

        eigval = eigval_all[ind]
        eigvec = eigvec_all[:, ind]

        ind = np.argmax(abs(eigvec))
        self.ek = np.zeros(ndof)
        self.ek[ind] = 1.0

        eigval_real = np.real(eigval)
        eigval_imag = np.imag(eigval)
        eigvec_real = np.real(eigvec)
        eigvec_imag = np.imag(eigvec)

        # Fill in the top level state variable.
        self.v[0:ndof] = eigvec_real[:]
        self.v[ndof:2 * ndof] = eigvec_imag[:]
        self.v[2 * ndof] = eigval_real
        self.v[2 * ndof + 1] = eigval_imag

        return self.v

    def compute_int(self, key):

        f_int = self.f_int_dict[key]
        val = f_int(self.v, self.x)

        self.f_int_val_dict[key] = val

        return val

    def set_adjoint_coeff(self):

        # Extract variables
        ndof = self.ndof
        w = self.w
        x = self.x
        f_Jacobian = self.f_Jacobian

        # Extract the top level state variable
        eigvec_real = np.zeros(ndof)
        eigvec_imag = np.zeros(ndof)

        eigvec_real[:] = self.v[0:ndof]
        eigvec_imag[:] = self.v[ndof:2 * ndof]
        eigval_real = self.v[2 * ndof]
        eigval_imag = self.v[2 * ndof + 1]

        # Assemble adjoint coefficient matrix
        Jacobian = f_Jacobian(w, x)
        preshat_pv_T = np.zeros((2 * ndof + 2, 2 * ndof + 2))

        # 1st row
        preshat_pv_T[0:ndof, 0:ndof] = Jacobian - eigval_real * np.eye(ndof)
        preshat_pv_T[0:ndof, ndof:2 * ndof] = eigval_imag * np.eye(ndof)
        preshat_pv_T[0:ndof, 2 * ndof] = - eigvec_real[:]
        preshat_pv_T[0:ndof, 2 * ndof + 1] = eigvec_imag[:]

        # 2nd row
        preshat_pv_T[ndof:2 * ndof, 0:ndof] = - eigval_imag * np.eye(ndof)
        preshat_pv_T[ndof:2 * ndof, ndof:2 * ndof] = Jacobian - eigval_real * np.eye(ndof)
        preshat_pv_T[ndof:2 * ndof, 2 * ndof] = - eigvec_imag[:]
        preshat_pv_T[ndof:2 * ndof, 2 * ndof + 1] = - eigvec_real[:]

        # 3rd row
        preshat_pv_T[2 * ndof : 2 * ndof + 1, 0:ndof] = 2 * eigvec_real[:]
        preshat_pv_T[2 * ndof : 2 * ndof + 1, ndof:2 * ndof] = 2 * eigvec_imag[:]

        # 4th row
        preshat_pv_T[2 * ndof + 1 : , ndof:2 * ndof] = 2 * self.ek[:]

        preshat_pv_T = preshat_pv_T.T

        self.preshat_pv_T = preshat_pv_T

    def set_adjoint_RHS(self, key):

        '''
            Set the RHS of the adjoint equation.
        '''

        f_pint_pv = self.f_pint_pv_dict[key]
        pint_pv = f_pint_pv(self.v, self.x)
        self.pint_pv_dict[key] = pint_pv

    def solve_adjoint(self, key):

        '''
            Solve the adjoint equation.
        '''

        self.set_adjoint_coeff()
        self.set_adjoint_RHS(key)

        preshat_pv_T = self.preshat_pv_T
        pint_pv = self.pint_pv_dict[key]

        phi = np.linalg.solve(preshat_pv_T, pint_pv)

        self.phi_dict[key] = phi

    def get_total_der(self, key):

        return self.dint_dJacobian_dict[key]

    def compute_total_der(self, key):

        # HACK:
        # Assume that:
        # f_pint_px_dict = 0
        # f_pint_pJacobian_dict = 0

        # Extract variables
        phi = self.phi_dict[key]
        v = self.v
        ndof = self.ndof

        qr = np.zeros(ndof)
        qi = np.zeros(ndof)
        phi1 = np.zeros(ndof)
        phi2 = np.zeros(ndof)

        # Extract eigenvector and eigenvalue
        qr[:] = v[0:ndof]
        qi[:] = v[ndof:2 * ndof]

        # Extract eigenvector and eigenvalue
        phi1[:] = phi[0:ndof]
        phi2[:] = phi[ndof:2 * ndof]

        self.dint_dJacobian_dict[key] = [phi1, qr, phi2, qi]

        return [phi1, qr, phi2, qi]
