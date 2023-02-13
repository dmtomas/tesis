import numpy as np
import matplotlib.pyplot as plt
from AI import AI
import pandas as pd
import seaborn as sns

abrir = np.array(pd.read_csv("optimo.txt"))
optimo = []
for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])
states = [-np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
results = []
results2 = []
observation = [0 for i in range(42)]
observation2 = [0 for i in range(42)]
for i in range(len(states[0])):
    bot = AI(0.01, 0.05, [0, 0, 0], 1, [states[0][i], states[1][i]], 0.001)
    bot2 = AI(0.01, 0.05, [0.6, 0.35, 0.27, 0.144], 1, [states[0][i], states[1][i]], 0.0001)
    action, theta = bot.Policy(observation)
    action2, theta = bot2.Policy(observation)
    observation[-2] = action[0]
    observation[-1] = action[1]
    observation2[-2] = action2[0]
    observation2[-1] = action2[1]
    results.append(bot.Predicted_Reward(observation) / 2 + 0.5)
    results2.append(bot2.Predicted_Reward(observation2) / 2 + 0.5)

sns.set_theme(style="whitegrid")
plt.plot(np.linspace(0, 1, 100), optimo, label="Optimo numérico", linestyle="--", color="#D81B60")
plt.plot(np.linspace(0, 1, 100), 1 - 1/2 * np.e**(-2*(np.linspace(0, 1, 100))**2), label="Kennedy", color="#FFC107")
plt.plot(np.linspace(0, 1, 100), results2, label="Optimizado IA", color="#1E88E5")
plt.xlabel("Intencidad")
plt.ylabel("Probabilidad de exito")
plt.ylim(0.5, 1)
plt.xlim(0.01, 1)
plt.legend()
plt.show()
