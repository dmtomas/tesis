import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

abrir = np.array(pd.read_csv("reward_no_noise.csv"))
optimo = []

for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])

sns.set_theme(style="whitegrid")
x = [i for i in range(len(optimo))]
plt.plot(x, optimo, label="Predicción AI", color="#1E88E5")
plt.plot(x, [0.7226481667398055 for i in range(len(x))], linestyle="--", label="Valor Teórico", color="#D81B60")
plt.xlabel("Intervalos")
plt.ylabel("Recompensa")
plt.xlim(0, 5621)
plt.ylim(-1, 1)
plt.legend()
plt.show()