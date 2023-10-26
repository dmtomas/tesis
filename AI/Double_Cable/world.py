from pennylane import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import gym
from gym import spaces
import csv

class Enviroment_double(gym.Env):
    def __init__(self):
        super(Enviroment_double, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1,
                                            shape=(8,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(11,), dtype=np.float32)
        self.states = [0, 0]

    def reset(self):
        #self.states[0] = -np.random.uniform(0, 0.5)
        #self.states[1] = -self.states[0]
        observation = []
        self.done = False
        self.steps = 5
        self.a = [np.abs(self.states[0]), np.abs(self.states[0]), 0, 0, 0, 0, 0, 0]
        correct_0 = self.correr(0, self.a)
        correct_1 = self.correr(1, self.a)
        for i in range(0, 20):
            if self.correr(0, self.a) == 0:
                correct_0 += 1 / 40
            if self.correr(1, self.a) == 1:
                correct_1 += 1 / 40

        observation = [correct_0, correct_1, np.abs(self.states[0])] + self.a
        return observation
    
    def step(self, action):
        self.steps -= 1
        self.a = action
        correct_0 = 0
        correct_1 = 0
        for i in range(0, 20):
            if self.correr(0, self.a) == 0:
                correct_0 += 1 / 40
            if self.correr(1, self.a) == 1:
                correct_1 += 1 / 40

        # The observation is how many correct of each other where guesed in the previous atempt 
        # with parameters self.a, and a state with |\alpha|.
        observation = [correct_0, correct_1, np.abs(self.states[0])] + list(self.a)
        self.reward = correct_0 + correct_1

        if self.steps == 0: # Save the final reward when the steps are finished.
            save_data = [correct_0 + correct_1]
            b = open("Double_Cable/Data/Run.csv", "a+", newline="")
            writer = csv.writer(b)
            writer.writerow(save_data)
            self.done = True
        info = {}

        return observation, self.reward, self.done, info
    
    def correr(self, selec, action):
        a = self.states
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
        prog = sf.Program(2)
        with prog.context as q:
            # State preparation in Blackbird
            Coherent(a[selec], 0) | q[0]
            Coherent(a[selec], 0) | q[1]
            Dgate(action[0], 0) | q[0]
            Dgate(action[1], 0) | q[1]
            BSgate(action[2] * 2 * np.pi, action[3] * 2 * np.pi) | (q[0], q[1])
            Sgate(action[4], action[5] * 2 * np.pi) | q[0]
            Sgate(action[6], action[7] * 2 * np.pi) | q[1]
            MeasureFock() | (q[0], q[1])

        results = eng.run(prog)
        if results.samples[0][0] + results.samples[0][1] > 0:
            ans = 1
        if results.samples[0][0] + results.samples[0][1] == 0:
            ans = 0
        return ans
