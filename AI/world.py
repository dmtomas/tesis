from pennylane import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from Plots.AI import AI
import csv

class Enviroment:
    def __init__(self, states):
        self.states = states
    
    def correr(self, angle, selec, inten):
        a = self.states
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
        prog = sf.Program(1)
        with prog.context as q:
            # State preparation in Blackbird
            Coherent(a[selec], 0) | q[0]
            Dgate(inten, angle) | q[0]
            #LossChannel(1) | q[0]
            MeasureFock() | q[0]
        results = eng.run(prog)
        return results.samples[0][0]

    def reset(self):
        photon = []
        observation = []
        self.done = False
        self.steps = 10
        self.a = [np.abs(self.states[0]), -np.angle(self.states[0])]  # Angle and displacement
        for i in range(0, 20):
            photon.append(self.correr(self.a[1], 0, self.a[0]))
        for i in range(0, 20):
            photon.append(self.correr(self.a[1], 1, self.a[0]))
        ans = [0, 0]
        ans2 = [0, 0]
        for i in range(0, 20):
            ans[0] += photon[i] / 20
            ans2[0] += photon[i] ** 2 / 20
        for i in range(20, 40):
            ans[1] += photon[i] / 20
            ans2[1] += photon[i] ** 2 / 20
        for i in range(0, 2):
            ans2[i] -= ans[i] ** 2
        observation += ans + ans2 + self.a
        return observation, self.done
    
    def step(self, action):
        photon = []
        observation = []
        self.a = action
        for i in range(0, 20):
            photon.append(self.correr(self.a[1], 0, self.a[0]))
        for i in range(0, 20):
            photon.append(self.correr(self.a[1], 1, self.a[0]))
        ans = [0, 0, 0]
        ans2 = [0, 0, 0]
        for i in range(0, 20):
            ans[0] += photon[i] / 20
            ans2[0] += photon[i] ** 2 / 20
            if photon[i] == 0:
                ans[2] += 1
        for i in range(20, 40):
            ans[1] += photon[i] / 20
            ans2[1] += photon[i] ** 2 / 20
            if photon[i] == 0:
                ans2[2] += 1
        for i in range(0, 2):
            ans2[i] -= ans[i] ** 2
        observation += ans + ans2 + self.a
        """
        self.reward = 0
        # Can make function but this is better for noissy.
        for i in range(0, 20):
            if photon[i] == 0:
                self.reward += 0.025
            else:
                self.reward -= 0.025
        for i in range(20, 40):
            if photon[i] > 0:
                self.reward += 0.025
            else:
                self.reward -= 0.025
        """
        disp = complex(np.cos(action[1]) * action[0], np.sin(action[1]) * action[0])
        self.reward = -np.e**(-np.abs(self.states[1] + disp)**2) + np.e**(-np.abs(self.states[0] + disp)**2)
        if self.steps == 0:
            self.done = True
        self.steps -= 1
        return observation, self.reward, self.done

observation = []
reward = 0
theta = []
done = False
states = [-0.5, 0.5]
env = Enviroment(states)
bot = AI(0.01, 0.05, [0, 0, 0, 0], 1, states, 0.2)  # alpha, gamma, theta, delta, states
bot.epsilon = np.abs(bot.states[0] - bot.states[1])/2
for i in range(0, 5621):
    observation, done = env.reset()
    thetas = []
    bot.rewards = []
    bot.temporary = []
    bot.states[0] = -np.random.uniform(0, 1)
    bot.states[1] = -bot.states[0]
    env.states = [bot.states[0], bot.states[1]]
    while not done:
        action, theta = bot.Policy(observation)
        observation, reward, done = env.step(action)
        thetas.append(theta)
        bot.rewards.append(reward)
        bot.temporary.append(observation)
    theta = bot.Deep_Update(bot.temporary, thetas)

    print(str(action) + " " + str(reward) + " " + str(theta) + " " + str(bot.states))
    bot.theta = theta

    b = open("reward_no_noise.csv", "a+", newline="")
    writer = csv.writer(b)
    bot.states = [-0.5, 0.5]
    env.states = [bot.states[0], bot.states[1]]
    action, theta = bot.Policy(observation)
    observation, reward, done = env.step(action)
    writer.writerow([reward])
