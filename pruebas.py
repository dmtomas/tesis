import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.gaussian", wires=1, shots=1)

@qml.qnode(dev)
def mean_photon_gaussian(a, c, b_a, b_i):
    qml.CoherentState(a[c], 0, wires=0)
    qml.Displacement(b_i, b_a, wires=0)
    return qml.expval(qml.NumberOperator(0))

states = [-3, 0, 3]
values = []
for j in range(0, 10):
    for i in range(0, len(states)):
        values += [mean_photon_gaussian(states, i, 0, 3) for j in range(0, 20)]