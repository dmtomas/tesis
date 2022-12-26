import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import csv

N = 5
state = 1
total = 5000

eng = sf.RemoteEngine("simulon_gaussian")
prog = sf.Program(N)
with prog.context as q:
    # State preparation in Blackbird
    Coherent(state, 0) | q[0]
    for i in range(0, N-1):
        BSgate(np.pi/2, 0) | (q[i], q[i+1])
    MeasureFock() | q[i+1]

results = eng.run(prog, shots=total)
resultados = results.samples

with open('Transporte_simulado.csv', 'a+', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(resultados)