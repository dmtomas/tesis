import numpy as np
import scipy.optimize as sp
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def Real_observation(b, phi):
        return 1 - ((0.5 * np.e**(-(np.abs(-intensity + b[0] + complex(0, phi)))**2) + 0.5 * (1 - np.e**(-(np.abs(intensity + b[0] + complex(0, phi)))**2))))


steps = 100
intensity = 5

bounds = [(0, 10)]
a = sp.dual_annealing(Real_observation, bounds=bounds, args=(0,))
malo = a.x
print(malo)
observations = []
sin_modelo = []

phi = np.linspace(0, 2, steps)
for i in range(0, steps):
    bounds = [(-10, 10)]
    a = sp.dual_annealing(Real_observation, bounds=bounds, args=(phi[i],))
    observations.append(1 - a.fun)
    sin_modelo.append(1 - Real_observation(malo, phi[i]))


sns.set_style(style="whitegrid")
plt.plot(phi, sin_modelo, label="Sin modelo")
plt.plot(phi, observations, label="Con modelo")
plt.xlabel("Intensidad de las cuentas oscuras")
plt.ylabel("Probabilidad de éxito")
plt.legend()
plt.show()