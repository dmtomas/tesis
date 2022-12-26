import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import csv
from strawberryfields.tdm import borealis_gbs, get_mode_indices

total = 100
eng = sf.RemoteEngine("simulon_gaussian")
modes = 9

prog = sf.TDMProgram(modes, name="Tesis_try")
with prog.context() as q:
    # State preparation in Blackbird
    Coherent(-2, 0) | q[0]
    Coherent(-1.5, 0) | q[1]
    Coherent(-1, 0) | q[2]
    Coherent(-0.5, 0) | q[3]
    Coherent(0, 0) | q[4]
    Coherent(0.5, 0) | q[5]
    Coherent(1, 0) | q[6]
    Coherent(1.5, 0) | q[7]
    Coherent(2, 0) | q[8]
    MeasureFock() | (q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8])

results = eng.run_async(prog, shots=total, crop=True)
resultados = results.samples

with open('Valores_Reales.csv', 'a+', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(resultados)