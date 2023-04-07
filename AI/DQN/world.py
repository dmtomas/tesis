from pennylane import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from Neural_AI import AI
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
        ans = [0, 0, 0]
        ans2 = [0, 0, 0]
        for i in range(0, 20):
            ans[0] += photon[i] / 20
            ans2[0] += photon[i] ** 2 / 20
            if photon[i] == 0:
                ans[2] += 1/20
        for i in range(20, 40):
            ans[1] += photon[i] / 20
            ans2[1] += photon[i] ** 2 / 20
            if photon[i] == 0:
                ans2[2] += 1/20
        for i in range(0, 2):
            ans2[i] -= ans[i] ** 2
        observation = ans + ans2 + self.a
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
                ans[2] += 1/20
        for i in range(20, 40):
            ans[1] += photon[i] / 20
            ans2[1] += photon[i] ** 2 / 20
            if photon[i] == 0:
                ans2[2] += 1/20
        for i in range(0, 2):
            ans2[i] -= ans[i] ** 2
        observation = ans + ans2 + self.a
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
saved_dqn = "DQN/saved_dqn.csv"
bot = AI(0.1, [8, 6, 4, 2], [10, 6, 4, 1])  # alpha, policy, reward
bot.LoadAI(saved_dqn)
bot.temporary = []
z = 0
bot.rewards = []
print("The program has started")
while True:
    env.states[1] = np.random.uniform(0, 1)
    env.states[0] = -env.states[1]
    observation, done = env.reset()
    while not done:
        action, prediction = bot.Policy(observation)
        action[0] = np.abs(np.random.normal(action[0], np.sqrt(action[0])/4))
        observation, reward, done = env.step(action)
        while len(bot.temporary) >= 300:
            bot.temporary.pop(np.random.randint(0, len(bot.temporary)))
        bot.temporary.append([observation + action, reward])
    if len(bot.temporary) == 300:
        z += 1
        theta = bot.Deep_Update(bot.temporary)
        print(f"{action} {prediction} {reward} {env.states[1]} {bot.alpha}")

        if z % 5 == 0:
            z -= 5
            bot.states = [-0.5, 0.5]
            env.states = bot.states
            observation, done = env.reset()
            while not done:
                action, prediction = bot.Policy(observation)
                observation, reward, done = env.step(action)
            b = open("medidas_0,5.csv", "a+", newline="")
            writer = csv.writer(b)
            writer.writerow([action[0], bot.algo_quiero_guardar, bot.algo_quiero_guardar_2])
            b.close()
            bot.SaveAI(f"AI_step.csv")

    bot.theta = theta