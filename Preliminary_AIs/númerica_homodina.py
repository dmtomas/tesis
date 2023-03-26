import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.gaussian", wires=2, shots=1)

@qml.qnode(dev)
def mean_photon_gaussian():
    qml.CoherentState(10, 0, wires=0)
    qml.Squeezing(0.5, np.pi/3, wires=1)
    qml.Beamsplitter(np.pi/3, np.pi/2, wires=[0, 1])
    return qml.expval(qml.NumberOperator(0) - qml.NumberOperator(1))

print(mean_photon_gaussian())