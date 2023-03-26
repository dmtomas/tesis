import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

abrir = np.array(pd.read_csv("Plots/optimo_int_0.5_abs.txt"))
optimo = []
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])

x = np.linspace(0, 1, 100)
y = []
colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]

plt.rcParams['text.usetex'] = True

state = 0.5
x2 = np.linspace(0.01, 1, 101)

sns.set_theme(style="whitegrid")
plt.plot(x2[:-1], optimo[:-1], label="Desfasaje optimo", color="#D81B60")
plt.plot(x2, [0.5 for i in range(len(x2))], label="Desfasaje Kennedy", color="#FFC107")
plt.xlabel("Probabilidad de absorción")
plt.ylabel("Desplazamiento")
plt.xlim(0.01, 0.99)
plt.legend()
plt.show()