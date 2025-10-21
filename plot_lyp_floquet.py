import numpy as np
import matplotlib.pyplot as plt

folder = 'sweep/mbar_sweep/'

k5_vals = np.linspace(5, 13, 2)
mult_vals = np.zeros_like(k5_vals, dtype=np.complex128)
l_vals = np.zeros_like(k5_vals)

for i in range(len(k5_vals)):
    k5 = k5_vals[i]
    data = np.loadtxt(folder + f'flo_l_mu_mbar_{k5:.2f}.csv', dtype=np.complex128, delimiter=',')
    mult_vals[i] = np.max((np.abs(data[:-2])))
    l_vals[i] = data[4].real 

fig, ax = plt.subplots(2, 1, figsize=(8, 10))
ax[0].plot(k5_vals, mult_vals, marker='o')
ax[1].plot(k5_vals, l_vals, marker='o', color='orange')
plt.show()

