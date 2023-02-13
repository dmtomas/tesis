import numpy as np
import pandas as pd
abrir = np.array(pd.read_csv("optimo.txt"))
optimo = []

for i in range(0, len(abrir)):
    optimo.append(abrir[i][0])
maximo = 0
x = np.linspace(0, 2, 101)
for i in range(0, len(x)):
    a = 1/2 + 1/2 * (1-np.e**(-2*(np.abs(x[i])**2)))
    b = optimo[i]
    if maximo < b/a:
        maximo = b/a

print(maximo)