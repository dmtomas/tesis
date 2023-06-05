from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
import numpy as np


matrix = [[1, 0],
          [0, 1]]
gate = UnitaryGate(matrix)

circuit = QuantumCircuit(2) 

tq = QuantumRegister(1, "alpha")
tc0 = ClassicalRegister(1, "tc0")


circuit.append(gate, [0])

teleport = QuantumCircuit(tq, tc0)
teleport.h(tq[0])



teleport.measure(tq[0], tc0[0])
teleport.draw(output='latex', scale=5,filename='my_circuit.png')