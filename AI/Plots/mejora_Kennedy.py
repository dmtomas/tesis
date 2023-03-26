import numpy as np
import pandas as pd
from AI import AI

abrir = np.array(pd.read_csv("Plots/optimo.txt"))
optimo = []

for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])

x = np.linspace(0.01, 1, len(optimo))

bot = AI(0.01, 0.05, [0.73014566, 0.24268174, 0.09902827, 0.09862708], 1, [0, 0], 0.0001)

ans = 0
for i in range(0, len(x)):
    observation = [0, 0, 0, 0, 0, 0, 0, 0]
    bot.states = [-x[i], x[i]]
    action, theta = bot.Policy(observation)
    ans += np.abs(action[0])/(len(x) * optimo[i])
print(ans)