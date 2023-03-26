import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special
import seaborn as sns


def Helstrom(alpha):
    return 1/2 * (1 + np.sqrt(1 - np.e**(-np.abs(2 * alpha)**2)))

def Homodine(alpha):
    return 1/2 * (1 + special.erf(alpha))

abrir = np.array(pd.read_csv("Plots/optimo2.txt"))
optimo = []
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])

states = np.linspace(0, 2, len(optimo))


sns.set_theme(style="whitegrid")
plt.plot(states, optimo, label="Kennedy Optimizado", color="#004D40")
plt.plot(states, Helstrom(states), label="Limite de Helstrom", color="#D81B60")
plt.plot(states, Homodine(states), label="Homodina", color="#FFC107", linestyle="--")
plt.plot(states, (1 - 1/2 * np.e**(-2*(states)**2)), label="Kennedy", color="#1E88E5")
plt.xlabel(r"$\alpha$")
plt.ylabel("Probabilidad de éxito")
plt.xlim(0.01, 1)
plt.ylim(0.5, 1)
plt.legend()
plt.show()


