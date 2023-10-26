import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def recompenza(beta, state):
    return 0.5 * np.e**(-(np.abs(beta - state)**2)) + 0.5 * (1 - np.e**(-(np.abs(beta + state)**2)))

def Helstrom(alpha):
    return 1/2 * (1 + np.sqrt(1 - np.e**(-np.abs(2 * alpha)**2)))

abrir = np.array(pd.read_csv("Plots/optimo2.txt"))
optimo = []
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])

states_AI = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
disp_AI = np.array([0.72, 0.63, 0.7, 0.68, 0.8, 0.74, 0.82, 0.94, 0.98, 1])

sns.set_theme(style="whitegrid")
states = np.linspace(0, 2, len(optimo))
plt.scatter(states_AI, recompenza(states_AI, disp_AI), label="AI Solution", color="#D81B60", s=15, zorder=1)
plt.plot(states, optimo, label="Model Aware", color="#1E88E5", zorder=0)
plt.xlabel(r"$\alpha$")
plt.ylabel("Success probability")
plt.xlim(0, 1.05)
plt.ylim(0.5, 1)
plt.legend()
plt.show()