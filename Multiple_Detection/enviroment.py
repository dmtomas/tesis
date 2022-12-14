import pennylane as qml
from pennylane import numpy as np
import gym
from gym import spaces

def FindN(state, seleccionado):
    superior = 0
    if seleccionado == 0:
        superior = 0
    elif seleccionado != len(state) - 1:
        superior = (state[seleccionado] - state[seleccionado + 1]) / np.log(state[seleccionado] / state[seleccionado + 1])
    else:
        superior = np.infty
    return superior

dev = qml.device("default.gaussian", wires=1, shots=1)

@qml.qnode(dev)
def mean_photon_gaussian(a, c, b_a, b_i):
    qml.CoherentState(a[c], 0, wires=0)
    qml.Displacement(b_i, b_a, wires=0)
    return qml.expval(qml.NumberOperator(0))


class CustomEnv(gym.Env):
    def __init__(self, states):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low = -states[0]/2, high = -states[0] * 2, shape = (2,), dtype = np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty,
                                            shape=(20 * len(states) + 2,), dtype=np.float32)
        
        self.states = states

    def reset(self):
        self.steps = 10
        self.done = False
        self.angle = 0
        self.intencidad = -self.states[0]  # Valor teórico.
        self.limites = [FindN(self.states + np.array([self.intencidad for j in range(len(self.states))]), i) for i in range(0, len(self.states))]
        self.values = [] # Datos para calcular el ángulo.
        for i in range(0, len(self.states)):
            self.values += [mean_photon_gaussian(self.states, i, self.angle, self.intencidad) for j in range(0, 20)]
        self.observation = np.array(self.values + [self.angle, self.intencidad])
        return self.observation 


    def step(self, action):
        self.angle = action[0] / self.states[-1] * np.pi * 2
        self.intencidad = action[1]
        self.limites = [FindN(self.states + np.array([self.intencidad for j in range(len(self.states))]), i) for i in range(0, len(self.states))]
        self.values = []
        for i in range(0, len(self.states)):
            self.values += [mean_photon_gaussian(self.states, i, self.angle, self.intencidad) for j in range(0, 20)]
        self.observation = np.array(self.values + [self.angle, self.intencidad])

        self.reward = 0
        for i in range(0, len(self.values)):
            votar = 0
            for j in range(0, len(self.limites)): # Encontrar donde cae se puede hacer con binary search. Innecesario por ahora.
                if self.limites[j] >= np.round(self.values[i]):
                    votar = j
                    break
            if votar == i // 20:
                self.reward += 0.1
            else:
                self.reward -= 0.05
        print(self.reward)
        self.steps -= 1
        if self.steps == 0:
            self.done = True
        info = {}
        return self.observation, self.reward, self.done, info
