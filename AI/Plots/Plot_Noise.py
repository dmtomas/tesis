import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

abrir = np.array(pd.read_csv("Plots/optimo_0.5_abs.txt"))
abrir2 = np.array(pd.read_csv("Plots/optimo_1_abs.txt")) 
optimo = []
optimo2 = []
for i in range(0, len(abrir)):
    optimo2.append(abrir2[i][0])
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])
def absorcion(state, p):
    ans = 0
    for i in range(0, 20):
        ans += p**(i) * np.abs(state)**(2*i)/np.math.factorial(i)
    return 1 - 0.5 * np.e**(-np.abs(state)**2) * ans

x = np.linspace(0, 1, 100)
y = []
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]
sns.set_theme(style="whitegrid")
plt.rcParams['text.usetex'] = True

state = 0.5
x2 = np.linspace(0.01, 1, 101)
plt.plot(x2, optimo, color=colors[1], label=r"Optimizado $|\alpha| = 0.5$", linestyle="--")
plt.plot(x2, optimo2, color=colors[4], label=r"Optimizado $|\alpha| = 1$", linestyle="--")
plt.plot(x, absorcion(2 * 0.5, x), label=r"Kennedy $|\alpha| = 0.5$", color=colors[0])
plt.plot(x, absorcion(2, x), color=colors[3], label=r"Kennedy $|\alpha| = 1$")

plt.xlabel("Probabilidad de absorción")
plt.ylabel("Probabilidad de exito")
plt.xlim(0, 1)
plt.ylim(0.5, 1)
plt.legend()
plt.show()