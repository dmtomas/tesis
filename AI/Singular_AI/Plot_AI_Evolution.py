import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

for j in range(1, 11):
    abrir = np.array(pd.read_csv(f"Singular_AI/Data/reward_no_noise_{np.round(j/10, 2)}.csv"))
    optimo = []
    for i in range(0, len(abrir)):
        optimo.append(abrir[i][1])
    sns.set_theme(style="whitegrid")
    steps = np.linspace(0, len(optimo), len(optimo))
    plt.plot(steps, optimo, label=r"$|\alpha|=$"+str(np.round(j/10, 2)), zorder=0)
plt.xlabel("Pasos en entrenamiento")
plt.ylabel("Probabilidad de éxito")
plt.ylim(0.6, 1.1)
plt.xlim(0, 1000)
plt.legend()
plt.show()