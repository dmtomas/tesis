from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

tq = QuantumRegister(3, "alpha")
tc0 = ClassicalRegister(1, "tc0")
tc1 = ClassicalRegister(1, "tc1")
tc2 = ClassicalRegister(1, "tc2")


teleport = QuantumCircuit(tq, tc0, tc1, tc2)
teleport.h(tq[1])
teleport.cx(tq[1], tq[2])



teleport.measure(tq[2], tc2[0])
teleport.draw(output='latex', scale=5,filename='my_circuit.png')