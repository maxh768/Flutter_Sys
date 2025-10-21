import numpy as np
import matplotlib.pyplot as plt
import niceplots

dt = 0.005
T0 = 500
Tf = 500 + (9 * dt)
t = np.arange(T0, Tf, dt)
print(t)
pert = np.linspace(0, 7 * np.pi/180.0, 5)
pert_2 = np.linspace(7 * np.pi/180.0, 0, 5)
pert = np.concatenate((pert, pert_2))

plt.style.use(niceplots.get_style('james-light'))

plt.figure(figsize=(12,6))
plt.plot(t, pert, marker='o')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$', rotation=0)
# plt.title('Perturbation in Theta over Time')
plt.grid()
plt.xticks(t)
plt.savefig("pert_theta_profile.pdf")
plt.show()