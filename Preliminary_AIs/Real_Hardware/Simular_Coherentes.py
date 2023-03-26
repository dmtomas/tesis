import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import csv

N = 2
total = 100

eng = sf.RemoteEngine("simulon_gaussian")
prog = sf.Program(N)
with prog.context as q:
    # State preparation in Blackbird
    Sgate(4, 0) | q[0]
    BSgate(np.pi/4, 0) | (q[0], q[1])
    MeasureFock() | q[0]

results = eng.run(prog, shots=total)
resultados = results.samples

with open('Coherentes_simulados.csv', 'a+', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(resultados)