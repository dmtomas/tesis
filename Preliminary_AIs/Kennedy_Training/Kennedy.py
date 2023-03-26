import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.gaussian", wires=2)

@qml.qnode(dev)
def mean_photon_gaussian(b_a, c):
    a = [-1, 1]
    qml.CoherentState(a[c], 0, wires=0)
    qml.CoherentState(0.5, 0, wires=1)
    qml.CubicPhase(1, wires=1)
    qml.Beamsplitter(0.35, np.pi/2 + 0.35, wires=[0, 1])
    qml.Displacement(1, b_a, wires=0)
    return qml.expval(qml.NumberOperator(0))

print(mean_photon_gaussian(0, 1))
"""
c = np.random.randint(0, 2)

correcto = 0
total = 0
for i in range(0, 100):
    a = int(mean_photon_gaussian(0, c))
    if  a == 0 and c == 0:
        correcto += 1
        total += 1
    elif a >= 1 and c == 1:
        correcto += 1
        total += 1
    else:
        total += 1
print(correcto / total)
"""