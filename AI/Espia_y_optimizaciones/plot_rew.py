import numpy as np
import scipy.optimize as sp
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def original_rew(b, t):
    return 0.5 * np.e**(-np.abs(-t + complex(b[0], b[1]))**2) + 0.5 * (1 - np.e**(-np.abs(t + complex(b[0], b[1]))**2))

def Helstrom(t):
    return 0.5 * (1 + np.sqrt(1 - np.e**(-4 * np.abs(t**2))))

def Real_observation(b, disp):
        return 0.5 * np.e**(-np.abs(-intensity * np.cos(disp[2]) + complex(disp[0], disp[1]) + complex(b[0], b[1]))**2) + 0.5 * (1 - np.e**(-np.abs(intensity * np.cos(disp[2]) + complex(disp[0], disp[1]) + complex(b[0], b[1]))**2))


steps = 100
intensity = 1.5

bounds = [(-2, 2), (-2, 2)]
a = sp.dual_annealing(original_rew, bounds=bounds, args=(intensity,))
desc, rews_original = list(a.x), 1-a.fun

observations = []
displacements = []
phi = np.linspace(0, np.pi/2, steps)
for i in range(0, steps):
    bounds = [(-2, 2), (-2, 2)]
    paso = [desc[0], desc[1], phi[i]]
    a = sp.dual_annealing(Real_observation, bounds=bounds, args=(paso,))
    observations.append(1 - a.fun)
    displacements.append(list(a.x))


sns.set_style(style="whitegrid")
plt.plot(phi, [rews_original for i in range(len(phi))], linestyle="--", label="Éxito esperado")
plt.plot(phi, observations, label="Éxito observado")
plt.plot(phi, Helstrom(intensity * np.sin(phi)), label="Éxito del espía")
plt.xlabel("Parámetro separador de haces")
plt.ylabel("Probabilidad de éxito")
plt.legend()
plt.show()

plt.plot(phi, [displacements[i][0] for i in range(len(phi))], label="Imaginario")
plt.plot(phi, [displacements[i][1] for i in range(len(phi))], label="Real")
plt.legend()
plt.show()