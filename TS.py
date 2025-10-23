import numpy as np

def timeder_mat(T, n):

    """
        Time derivative matrix
    """

    D = np.zeros((n,n))

    if (n%2==0):
        # even
        for i in range(n):
            for j in range(n):
                if (i != j):
                    D[i, j] = 0.5 * (-1)**(i-j) / np.tan(np.pi*(i - j) / n)

    else:
        # odd
        for i in range(n):
            for j in range(n):
                if (i != j):
                    D[i, j] = 0.5 * (-1)**(i - j) / np.sin(np.pi*(i - j) / n)

    D = D * (2.0 * np.pi) / T

    return D

def sort_N(N):
    
    """
        Sorting the sequence such that
            it will return things in the following order for FFT
                Ex 1: N=5
                    return 0,1,2,-2,1
                Ex 2: N=6
                    return -2,-1,0,1,2,3
    """

    N_vec = []
    N_vec.append(0)

    if (N%2 == 1):
        # odd
        N_half = (N - 1) / 2

        for i in range(N_half):
            N_vec.append(i + 1)
        for i in range(N_half):
            N_vec.append(i - N_half)

    else:
        # even
        N_half = N / 2

        for i in range(N_half):
            N_vec.append(i + 1)
        for i in range(N_half - 1):
            N_vec.append(i - N_half + 1)

    return N_vec

def TS_evaluate(coef, t, period):
    
    """
        Evaluate data at time "t" with time period "period" 
        with FFT coefficient "coef".
    """

    t_scaled = t / period * (2.0 * np.pi)

    N = len(coef)
    coef = coef / N # rescaled the coef

    N_vec = sort_N(N)

    y = 0.0
    for i in range(N):
        N_loc = N_vec[i]
        y += coef[i] * np.exp(N_loc * 1j * t_scaled)

    return np.real(y)


flag_inter_test = False
if flag_inter_test:
    # Interpolation test
    # y = sin(t), T=2pi, N = 3
    N = 3
    y_inter = np.zeros(N)
    # sampling
    for i in range(N):
        phase = np.float(i)/np.float(N)*(2.0*np.pi)

        y_inter[i] = np.sin(phase)
    # FFT
    coeff_list = np.fft.fft(y_inter)
    # evaluate
    NN = 100
    y_test = np.zeros(NN)
    for i in range(NN):
        phase = np.float(i)/np.float(NN)

        y_test[i] = TS_evaluate(coeff_list, phase, 1.0)

    import matplotlib.pyplot as plt 

    plt.plot(y_test, 'o')
    plt.show()


if (__name__ == '__main__'):
    # Differentiation test
    # y = sin(t), T=2pi, N = 3
    N = 10
    t_vec = (np.linspace(0,2.0*np.pi,N+1))[:N]
    y_vec = np.matrix(np.zeros((N,1)))
    ydot_analy_vec = np.matrix(np.zeros((N,1)))
    for i in range(N):
        t_loc = t_vec[i]
        y_vec[i,0] = np.sin(t_loc)
        ydot_analy_vec[i,0] = np.cos(t_loc)

    D = timeder_mat(2.0*np.pi,N)
    ydot_vec = D.dot(y_vec)

    err = abs(ydot_analy_vec-ydot_vec)
    print('++++++++++err is++++++++++', err)

    print("D", D)
    print("D^2", D.dot(D))


