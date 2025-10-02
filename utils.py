import numpy as np

def apply_A(A, q):

    avec = A.dot(q)

    return avec

def apply_A_fad(A, q, Adot, qdot):

    avec_dot = Adot.dot(q) + A.dot(qdot)

    return avec_dot

def apply_A_rad(A, q, avec_bar):

    Abar = np.outer(avec_bar, q)
    qbar = np.transpose(A).dot(avec_bar)

    return Abar, qbar

def apply_B(ndof, B, q1, q2):

    bvec = np.zeros(ndof, dtype=complex)

    for i in range(ndof):

        q1TBq2 = (B[i, :, :].dot(q2)).dot(q1)
        bvec[i] = q1TBq2

    return bvec

def apply_B_fad(ndof, B, q1, q2, Bdot, q1dot, q2dot):

    bvec_dot = np.zeros(ndof, dtype=complex)

    for i in range(ndof):

        q1TBq2_d = (Bdot[i, :, :].dot(q2)).dot(q1) + (B[i, :, :].dot(q2)).dot(q1dot) + (B[i, :, :].dot(q2dot)).dot(q1)
        bvec_dot[i] = q1TBq2_d

    return bvec_dot

def apply_B_rad(ndof, B, q1, q2, bvec_bar):

    Bbar = np.zeros((ndof, ndof, ndof), dtype=complex)

    q1bar = np.zeros(ndof, dtype=complex)
    q2bar = np.zeros(ndof, dtype=complex)

    for i in range(ndof):
        Bbar[i, :, :] += bvec_bar[i] * np.outer(q1, q2)

        q1bar += bvec_bar[i] * B[i, :, :].dot(q2)
        q2bar += bvec_bar[i] * B[i, :, :].T.dot(q1)

    return Bbar, q1bar, q2bar

def apply_C(ndof, C, q1, q2, q3):

    q2TCq3 = np.zeros((ndof, ndof), dtype=complex)
    cvec = np.zeros(ndof, dtype=complex)

    for i in range(ndof):
        for j in range(ndof):
            q2TCq3[i, j] = (C[i, j, :, :].dot(q3)).dot(q2)

    for i in range(ndof):
        cvec[i] = q2TCq3[i, :].dot(q1)

    return cvec

def apply_C_fad(ndof, C, q1, q2, q3, Cdot, q1dot, q2dot, q3dot):

    q2TCq3 = np.zeros((ndof, ndof), dtype=complex)
    q2TCq3_dot = np.zeros((ndof, ndof), dtype=complex)
    cvec_dot = np.zeros(ndof, dtype=complex)

    for i in range(ndof):
        for j in range(ndof):  
            q2TCq3[i, j] = (C[i, j, :, :].dot(q3)).dot(q2)
            q2TCq3_dot[i, j] = (Cdot[i, j, :, :].dot(q3)).dot(q2) + (C[i, j, :, :].dot(q3)).dot(q2dot) + (C[i, j, :, :].dot(q3dot)).dot(q2)

    for i in range(ndof):
        cvec_dot[i] = q2TCq3[i, :].dot(q1dot) + q2TCq3_dot[i, :].dot(q1)

    return cvec_dot

def apply_C_rad(ndof, C, q1, q2, q3, cvec_bar):

    # Primal
    q2TCq3 = np.zeros((ndof, ndof), dtype=complex)
    prod = np.zeros(ndof, dtype=complex)

    for i in range(ndof):
        for j in range(ndof):  
            q2TCq3[i, j] = (C[i, j, :, :].dot(q3)).dot(q2)

    for i in range(ndof):
        prod[i] = q2TCq3[i, :].dot(q1)

    # Derivative
    q2TCq3_bar = np.zeros((ndof, ndof), dtype=complex)

    Cbar = np.zeros((ndof, ndof, ndof, ndof), dtype = complex)

    q1bar = np.zeros(ndof, dtype = complex)
    q2bar = np.zeros(ndof, dtype = complex)
    q3bar = np.zeros(ndof, dtype = complex)

    for i in range(ndof):
        q2TCq3_bar[i, :] += cvec_bar[i] * q1
        q1bar += cvec_bar[i] * q2TCq3[i, :]

    for i in range(ndof):
        for j in range(ndof):  
            q2bar += q2TCq3_bar[i, j] * C[i, j, :, :].dot(q3)
            q3bar += q2TCq3_bar[i, j] * C[i, j, :, :].T.dot(q2)
            Cbar[i, j, :, :] += q2TCq3_bar[i, j] * np.outer(q2, q3)

    return Cbar, q1bar, q2bar, q3bar

def inner_prod(p, q):

    return np.conj(p).dot(q)

def inner_prod_fad(p, q, pdot, qdot):

    return np.conj(pdot).dot(q) + np.conj(p).dot(qdot)

def inner_prod_rad(p, q, prod_bar):

    p_bar = np.conj(prod_bar) * q
    q_bar = prod_bar * p

    return p_bar, q_bar
