import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def recompenza(beta, state):
    return 0.5 * np.e**(-(np.abs(beta - state)**2)) + 0.5 * (1 - np.e**(-(np.abs(beta + state)**2)))

def Helstrom(alpha):
    return 1/2 * (1 + np.sqrt(1 - np.e**(- 8 * np.abs(alpha)**2)))

abrir = np.array(pd.read_csv("Double_Cable/Doble_Compara_Helstrom.csv"))
abrir2 = np.array(pd.read_csv("Double_Cable/Optimized_double.csv"))
x = []
x2 = []
optimo = []
optimo2 = []

for i in range(0, len(abrir)):
    x.append(float(abrir[i][0]))
    optimo.append(float(abrir[i][1]))
for i in range(0, len(abrir2)):
    x2.append(float(abrir2[i][0]))
    optimo2.append(float(abrir2[i][1]))

states2 = np.linspace(x2[0], x2[-1], len(optimo2))
sns.set_theme(style="whitegrid")
states = np.linspace(x[0], x[-1], len(x))
plt.plot(states2, optimo2, label="30 veces")
plt.plot(states, Helstrom(states), label="Limite de Helstrom Doble", color="#D81B60")
plt.plot(x, optimo, label="100 veces", color="#004D40", zorder=0)
plt.xlabel(r"$\alpha$")
plt.ylabel("Probabilidad de éxito")
plt.xlim(0.05, 0.5)
#plt.ylim(0.5, 1.1)
plt.legend()
plt.show()