import numpy as np
import matplotlib.pyplot as plt
import csv

def distribu(state, n):
    return np.e**(-(state**2)) * (np.abs(state)**(2*n))/np.math.factorial(n)


N = 1000
teoricos = []
reales = []
state = 1
n = 0

for n in range(0, 15):
    for i in range(0, int(distribu(state, n) * N)):
        teoricos.append(n)
    n += 1
n = max(teoricos) - min(teoricos)
with open("Coherentes_simulados.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    reales.append(int(row[0]))


plt.hist(x=reales, bins=n, range=[0, n],rwidth=0.95, label="Reales", density=True, color="#CD5C5C")
plt.hist(x=teoricos, bins=n, rwidth=0.95, label="Teóricos", density=True, edgecolor = 'black', ls="--", color="white", alpha=0.2)
plt.xlabel("Cantidad de Fotones")
plt.ylabel("frecuencia")

plt.legend()
plt.show()