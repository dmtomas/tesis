import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

abrir = np.array(pd.read_csv("Double_Cable/Doble_Compara_Helstrom.csv"))
x = []
optimo = [[], [], [], [], [], []]
for i in range(0, len(abrir)):
    for j in range(2, len(abrir[i])):
        optimo[j-2].append(float(abrir[i][j]))

states = np.linspace(0.05, 0.746, len(optimo[0]))
sns.set_theme(style="whitegrid")
legends = [r"$\theta$", r"$z_1$", r"$z_2$", r"$\eta$", r"$\beta$", r"$\gamma$"]
colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#8E7E97", "#0F00FF"]
for i in range(0, len(optimo)):
    plt.plot(states, optimo[i], label=legends[i], color=colors[i])
plt.xlabel(r"$\alpha$")
plt.ylabel("Probabilidad de éxito")
plt.xlim(0.05, 0.746)
plt.ylim(0, 6)
plt.legend()
plt.show()