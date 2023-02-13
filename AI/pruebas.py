import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def function(t):
    c = np.e**(-2*(np.abs(t)**2))
    return 0.5 * (1+np.sqrt(1- np.abs(c)**2))


sns.set_theme(style="whitegrid")
x = np.linspace(0, 2, 101)
abrir = np.array(pd.read_csv("optimo.txt"))
optimo = []
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])
print(optimo)
y = function(x)
plt.plot(x, y - optimo, label="Helstrom - Kennedy opt", color="#D81B60")
plt.plot(x, y - (1/2 + 1/2 * (1-np.e**(-2*(np.abs(x)**2)))), label="Helstrom - Kennedy", color="#1E88E5")
plt.xlabel("Intencidad")
plt.ylabel("Probabilidad de exito")
plt.xlim(0, 2)

plt.legend()
plt.show()
