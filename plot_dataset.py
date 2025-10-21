import matplotlib.pyplot as plt
import niceplots
import numpy as np
filename = 'sweep/mbar_sweep/data_mbar_13.00'
data = np.loadtxt(f'{filename}.csv')
data_T4 = np.loadtxt(f'{filename}_pert_T4.csv')
flo_file = np.loadtxt('sweep/mbar_sweep/flo_l_mu_mbar_13.00.csv', dtype=np.complex128, delimiter=',')
T = flo_file[-1]

custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263', '#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
c1 = custom_colors[0]
c2 = custom_colors[1]
c3 = custom_colors[2]
c4 = custom_colors[3]
c5 = custom_colors[4]
c6 = custom_colors[5]

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

labelpad = 45
fontsize = 16

plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize

plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.fontset"] = "custom"

fig, ax = plt.subplots(3, 2, figsize=(12, 6), sharex=False)
plt.style.use(niceplots.get_style('james-light'))

ax[0, 0].set_xlim([499, 507])
ax[0, 0].plot(data[:,0], data[:,1], '-', label=r"$\bar{h}$", color=c1)
# ax[0, 0].set_xlabel(r"$t$", fontsize=20)
ax[0, 0].set_ylabel(r"$\bar{h}$", fontsize=18, rotation=0)
ax[0, 0].xaxis.set_visible(False)
ax[0, 0].axvline(500, color=c6, linestyle='--', linewidth=1.5, zorder=10)
ax[0, 0].set_title(r'Perturbation at $T=0$')

ax[1, 0].set_xlim([499, 507])
ax[1, 0].plot(data[:,0], data[:,2], '-', label=r"$\alpha$", color=c2)
ax[1, 0].set_ylabel(r"$\alpha$", fontsize=18, rotation=0)
ax[1, 0].axvline(500, color=c6, linestyle='--', linewidth=1.5, zorder=10)

ax[2, 0].plot(data[:,0], data[:,2], '-', color=c2)
ax[2, 0].set_xlabel(r"$t$", fontsize=20)
ax[2, 0].set_xlim([400, 600])
ax[2, 0].set_ylabel(r"$\alpha$", fontsize=18, rotation=0)
ax[2, 0].set_ylim([0.2, 0.4])
ax[2, 0].axvline(500, color=c6, linestyle='--', linewidth=1.5, zorder=10)


#################

ax[0, 1].set_xlim([499, 507])
ax[0, 1].plot(data_T4[:,0], data_T4[:,1], '-', label=r"$\bar{h}$", color=c1)
ax[0, 1].axvline(500 + (T/4), color=c6, linestyle='--', linewidth=1.5, zorder=10)
ax[0, 1].xaxis.set_visible(False)
ax[0, 1].set_title(r'Perturbation at $T=\frac{1}{4}$')

ax[1, 1].set_xlim([499, 507])
ax[1, 1].plot(data_T4[:,0], data_T4[:,2], '-', label=r"$\alpha$", color=c2)
ax[1, 1].axvline(500 + (T/4), color=c6, linestyle='--', linewidth=1.5, zorder=10)

ax[2, 1].plot(data_T4[:,0], data_T4[:,2], '-', color=c2)
ax[2, 1].set_xlabel(r"$t$", fontsize=20)
ax[2, 1].set_xlim([400, 600])
ax[2, 1].set_ylim([0.2, 0.4])
ax[2, 1].axvline(500 + (T/4), color=c6, linestyle='--', linewidth=1.5, zorder=10)


ax[0, 0].set_xlim([499, 507])
ax[0, 1].set_xlim([499, 507])
ax[1, 0].set_xlim([499, 507])
ax[1, 1].set_xlim([499, 507])
ax[2, 0].set_xlim([400, 600])
ax[2, 1].set_xlim([400, 600])


plt.savefig("unsteady_ae_pert_theta_first.pdf")