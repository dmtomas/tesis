import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special


def Helstrom(alpha):
    return 1/2 * (1 + np.sqrt(1 - np.e**(-np.abs(2 * alpha)**2)))

def Homodine(alpha):
    return 1/2 * (1 + special.erf(alpha))

def Kennedy(b):
        return 1 - (0.5 * np.e**(-np.abs(-intensity + np.abs(b[0]))**2) + 0.5 * (1 - np.e**(-np.abs(intensity + np.abs(b[0]))**2)))

"""
steps = 100
intensity = 0

observations = []

states = np.linspace(0, 1, steps)
for i in range(0, steps):
    intensity = i / (steps - 1)
    bounds = [(-1.5, 1.5)]
    a = sp.dual_annealing(Kennedy, bounds=bounds)
    observations.append(1 - a.fun)


sns.set_style(style="whitegrid")
plt.plot(states, observations, label="Kennedy optimizada", color="#004D40")
plt.plot(states, Helstrom(states), label="Límite de Helstrom", color="#D81B60")
plt.plot(states, Homodine(states), label="Homodina", color="#FFC107", linestyle="--")
plt.plot(states, (1 - 1/2 * np.e**(-4*(states)**2)), label="Kennedy", color="#1E88E5")
plt.xlabel(r"$\alpha$")
plt.ylabel("Probabilidad de éxito")
plt.xlim(0.01, 1)
plt.ylim(0.5, 1)
plt.legend()
plt.show()
"""

vals = np.linspace(-5, 5, 1000)
obs = []
intensity = 0.7071

for i in range(len(vals)):
     obs.append(1 - Kennedy([vals[i]]))

sns.set_style(style="whitegrid")
plt.plot(vals, obs, color="#1E88E5")

plt.xlabel(r"$\beta$")
plt.ylabel("Probabilidad de éxito")
plt.ylim(0.5, 1.1)
plt.legend()
plt.show()
