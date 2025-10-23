import numpy as np
import matplotlib.pyplot as plt
import niceplots

k_5_folder = 'sweep/k5_sweep'
k_3_folder = 'sweep/k3_sweep'
mbar_folder = 'sweep/mbar_sweep'

mbar_arr = np.linspace(5, 13, 20)
k_3_arr = np.linspace(-3, 1, 20)
k_5_arr = np.linspace(30, 70, 20)

mult_k3 = np.zeros((4, k_3_arr.shape[0]))
l_k3 = np.zeros_like(k_3_arr)

mult_mbar = np.zeros((4, mbar_arr.shape[0]))
l_mbar = np.zeros_like(mbar_arr)

mult_k5 = np.zeros((4, k_5_arr.shape[0]))
l_k5 = np.zeros_like(k_5_arr)

for i in range(len(k_5_arr)):
    k5 = k_5_arr[i]
    data = np.loadtxt(f'{k_5_folder}/flo_l_mu_k5_{k5:.2f}.csv', dtype=np.complex128, delimiter=',')
    flo_mult = data[:4]
    # flo_exp = np.log(flo_mult) / data[-1]
    mult_k5[:, i] = (np.abs(flo_mult))
    l_k5[i] = data[-3].real

for i in range(len(k_3_arr)):
    k3 = k_3_arr[i]
    data = np.loadtxt(f'{k_3_folder}/flo_l_mu_k3_{k3:.2f}.csv', dtype=np.complex128, delimiter=',')
    flo_mult = data[:4]
    # flo_exp = np.log(flo_mult) / data[-1]
    mult_k3[:, i] = (np.abs(flo_mult))
    l_k3[i] = data[-3].real

for i in range(len(mbar_arr)):
    mbar = mbar_arr[i]
    data = np.loadtxt(f'{mbar_folder}/flo_l_mu_mbar_{mbar:.2f}.csv', dtype=np.complex128, delimiter=',')
    flo_mult = data[:4]
    # flo_exp = np.log(flo_mult) / data[-1]
    mult_mbar[:, i] = (np.abs(flo_mult))
    l_mbar[i] = data[-3].real


mask_k3 = mult_k3 < 0.98
mask_mbar = mult_mbar < 0.98
mask_k5 = mult_k5 < 0.98
plt.style.use(niceplots.get_style('james-light'))

fig, ax = plt.subplots(2, 3, figsize=(15, 7))

# k3:
ax[0, 0].set_title(r'$k_3$ sweep')
max_under_one_k3 = np.where(mask_k3, mult_k3, -np.inf)
ax[0, 0].plot(k_3_arr, np.max(max_under_one_k3, axis=0), marker='o')
ax[0, 0].set_ylabel('Floquet Multiplier')
ax[0, 0].set_ylim([0, 1])

# ax[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax[1, 0].plot(k_3_arr, l_k3, marker='o', color='orange')
ax[1, 0].set_ylabel('First Lyapunov Coefficient')
ax[1, 0].set_xlabel(r'$k_3$')

# mbar:
ax[0, 1].set_title(r'$\bar{m}$ sweep')
max_under_one_mbar = np.where(mask_mbar, mult_mbar, -np.inf)
ax[0, 1].plot(mbar_arr, np.max(max_under_one_mbar, axis=0), marker='o')
ax[0, 1].set_ylim([0, 1])

# ax[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax[1, 1].plot(mbar_arr, l_mbar, marker='o', color='orange')
ax[1, 1].set_xlabel(r'$\bar{m}$')

# k5:
ax[0, 2].set_title(r'$k_5$ sweep')
max_under_one_k5 = np.where(mask_k5, mult_k5, -np.inf)
ax[0, 2].plot(k_5_arr, np.max(max_under_one_k5, axis=0), marker='o')
ax[0, 2].set_ylim([0, 1])

# ax[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.7)
ax[1, 2].plot(k_5_arr, l_k5, marker='o', color='orange')
ax[1, 2].set_xlabel(r'$k_5$')


plt.show()

