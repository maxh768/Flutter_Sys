import matplotlib.pyplot as plt
import niceplots
import numpy as np
data = np.loadtxt('sweep/k5_sweep/data_k5_30.00.csv')

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

fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
plt.style.use(niceplots.get_style('james-light'))

# start_ind = int(np.round((t_start_pert) * t_step - 100))
# end_ind = int(np.round((t_end_pert) * t_step + 100))
# start_ind_t = int(start_ind/t_step)
# end_ind_t = int(end_ind/t_step)

ax[0].plot(data[:,0], data[:,1], '-', label=r"$\bar{h}$", color=c1)
ax[0].set_xlabel(r"$t$", fontsize=20)
ax[0].set_ylabel(r"$\bar{h}$", fontsize=18, rotation=0)
# ax[0].set_xlim([start_ind,end_ind])
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_label_coords(-0.1,0.5)

ax[1].plot(data[:,0], data[:,2], '-', label=r"$\alpha$", color=c2)
ax[1].set_xlabel(r"$t$", fontsize=20)
ax[1].set_ylabel(r"$\alpha$", fontsize=18, rotation=0)
# ax[1].set_xlim([start_ind,end_ind])
ax[1].yaxis.set_label_coords(-0.1,0.5)

plt.savefig("unsteady_ae_pert_theta_first.pdf")