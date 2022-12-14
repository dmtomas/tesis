import pennylane as qml
from pennylane import numpy as np
import gym
from gym import spaces


dev = qml.device("default.gaussian", wires=1, shots=1)

@qml.qnode(dev)
def mean_photon_gaussian(b_a, c, b_i):
    a = [-0.5, 0.5]
    qml.CoherentState(a[c], np.random.uniform(0, 1.5), wires=0)
    #qml.CoherentState(1, 0, wires=1)
    #qml.Squeezing(1, np.pi/3, wires=1)
    #qml.Beamsplitter(np.random.uniform(-1, 1), np.pi/2 + np.random.uniform(-1, 1), wires=[0, 1])
    qml.Displacement(b_i, b_a, wires=0)
    return qml.expval(qml.NumberOperator(0))


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2,), dtype = np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.infty, high=np.infty,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        self.done = False
        self.total = 0
        self.correctos = 0
        self.steps = 100000
        self.seleccionado = np.random.randint(0, 2)
        self.angle = 0.5
        self.intencidad = 0.5
        self.observation = np.array([mean_photon_gaussian(self.angle, self.seleccionado, self.intencidad)])
        return self.observation 


    def step(self, action):
        self.steps -= 1
        self.total += 1
        self.seleccionado = np.random.randint(0, 2)
        self.angle = action[0] * np.pi
        self.intencidad = action[1] * 2
        self.observation = np.array([mean_photon_gaussian(self.angle, self.seleccionado, self.intencidad)])

        if int(self.observation[0]) == 0 and self.seleccionado == 0:
            self.correctos += 1
        elif int(self.observation[0]) >= 1 and self.seleccionado == 1:
            self.correctos += 1
        
        if self.steps <= 0:
            #self.reward = ((self.correctos / self.total) - 0.5) * 20
            self.done = True
            self.reward = 0
            #print(self.correctos / self.total)
        else:
            if int(self.observation[0]) >= 1 and self.seleccionado == 1:
                self.reward = 0.1
            elif int(self.observation[0]) == 0 and self.seleccionado == 0:
                self.reward = 0.1
            else:
                self.reward = -0.1
        info = {}
        return self.observation, self.reward, self.done, info

if __name__=="__main__":
    enviroment = CustomEnv()
    enviroment.reset()
    for i in range(0, 100):
        action = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1)])
        enviroment.step(action)