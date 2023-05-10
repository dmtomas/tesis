import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def recompenza(beta, state):
    return 0.5 * np.e**(-(np.abs(beta - state)**2)) + 0.5 * (1 - np.e**(-(np.abs(beta + state)**2)))

def Helstrom(alpha):
    return 1/2 * (1 + np.sqrt(1 - np.e**(- 8 * np.abs(alpha)**2)))

abrir = np.array(pd.read_csv("Double_Cable/Kennedy_Opt_Num.csv"))
x = []
optimo = []

for i in range(0, len(abrir)):
    x.append(float(abrir[i][0]))
    optimo.append(float(abrir[i][2]))

sns.set_theme(style="whitegrid")
states = np.linspace(x[0], x[-1], len(x))
plt.plot(states, Helstrom(states), label="Limite de Helstrom Doble", color="#D81B60")
plt.plot(states, 1 - 0.5 * np.e**(-8 * (states**2)), label="Kennedy Doble", color="#1E88E5")
plt.plot(x, optimo, label="Kennedy Doble optimizado", color="#004D40", zorder=0)
plt.xlabel(r"$\alpha$")
plt.ylabel("Probabilidad de éxito")
plt.xlim(0.05, 0.5)
plt.legend()
plt.show()