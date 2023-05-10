import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from timer import timer, get_timer
import time

@timer('function:gauss', unit='s')
def Gaussiana(action):
    eng = sf.Engine("gaussian")
    prog = sf.Program(2)

    with prog.context as q:
        Coherent(0.5, 0) | q[0]
        Coherent(0.5, 0) | q[1]
        Dgate(action[0], 0) | q[0]
        Dgate(action[1], 0) | q[1]
        BSgate(action[2] * 2 * np.pi, action[3] * 2 * np.pi) | (q[0], q[1])
        Sgate(action[4], action[5] * 2 * np.pi) | q[0]
        Sgate(action[6], action[7] * 2 * np.pi) | q[1]
        MeasureFock() | (q[0], q[1])

    result = eng.run(prog, shots=1000)

@timer
def Fock(action):
    a = []
    for i in range(0, 20):
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 6})
        prog = sf.Program(2)

        with prog.context as q:
            Coherent(0.5, 0) | q[0]
            Coherent(0.5, 0) | q[1]
            Dgate(action[0], 0) | q[0]
            Dgate(action[1], 0) | q[1]
            BSgate(action[2] * 2 * np.pi, action[3] * 2 * np.pi) | (q[0], q[1])
            Sgate(action[4], action[5] * 2 * np.pi) | q[0]
            Sgate(action[6], action[7] * 2 * np.pi) | q[1]
            MeasureFock() | (q[0], q[1])

        result = eng.run(prog)
        a.append(result.samples[0])

if __name__ == '__main__':
    action = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    with timer('time.sleep(2)') as t:
        Gaussiana(action)
        a = t.elapse
        print(f'after time.sleep(1) once, t.elapse = {t.elapse}')
        for i in range(0, 50):
            Fock(action)
        print(f'after time.sleep(1) twice, t.elapse = {t.elapse-a}')

