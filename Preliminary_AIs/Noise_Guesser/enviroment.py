import pennylane as qml
from pennylane import numpy as np
import gym
from gym import spaces

def Reward(action, ruido):
    ans = 0
    for i in range(0, len(action)):
        if (action[i] - ruido[i]) > 0.05:
            ans += 1/(action[i] - ruido[i])**2 / 400
        else:
            ans += 1
    return ans

dev = qml.device("default.gaussian", wires=3, shots=1)

@qml.qnode(dev)
def mean_photon_gaussian(c):
    a = [-0.5, 0.5]
    qml.CoherentState(a[c[0]], np.random.uniform(0, 1.5), wires=0)

    qml.CoherentState(c[1], c[2], wires=1)
    qml.Squeezing(c[3], c[4], wires=1)
    qml.Beamsplitter(c[5], np.pi/2 + c[6], wires=[0, 1])

    qml.CoherentState(c[1], c[2], wires=2)
    qml.Beamsplitter(c[7], np.pi/2 + c[8], wires=[0, 2])
    qml.Displacement(0.25, 0, wires=0)

    return qml.expval(qml.NumberOperator(0))


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low = -1, high = 1, shape = (8,), dtype = np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=np.infty,
                                            shape=(136,), dtype=np.float32)

    def reset(self):
        self.datos = "0101010001101111011011011110000101110011001000000111010001100101011100110110100101110011001000000101001001101111011000110110101101110011"
        self.done = False
        self.steps = 0
        self.seleccionado = 0
        self.ruido = [np.random.uniform(-1, 1) for i in range(0, 8)]
        self.observation = np.array([-1 for i in range(0, len(self.datos))])
        return self.observation

    def step(self, action):
        a = mean_photon_gaussian([self.seleccionado] + self.ruido)
        if a < 0:
            a = 0
        self.seleccionado = int(self.datos[self.steps])
        for i in range(0, len(self.datos)):
            if self.observation[i] == -1:
                self.observation[i] = a
                break
        self.reward = Reward(action, self.ruido)
        self.steps += 1
        if self.steps >= len(self.datos):
            self.done = True
        info = {}
        return self.observation, self.reward, self.done, info

if __name__=="__main__":
    enviroment = CustomEnv()
    enviroment.reset()
    for i in range(0, 100):
        action = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
        enviroment.step(action)