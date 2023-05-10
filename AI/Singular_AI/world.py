from pennylane import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from AI import AI
import csv

class Enviroment:
    def __init__(self, states):
        self.states = states
    
    def correr(self, selec, inten):
        a = self.states
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
        prog = sf.Program(1)
        with prog.context as q:
            # State preparation in Blackbird
            Coherent(a[selec], 0) | q[0]
            Dgate(inten, 0) | q[0]
            MeasureFock() | q[0]
        results = eng.run(prog)
        return results.samples[0][0]

    def reset(self):
        photon = []
        observation = []
        self.done = False
        self.steps = 10
        self.a = [np.abs(self.states[0]), -np.angle(self.states[0])]  # Angle and displacement
        return observation, self.done
    
    def step(self, action):
        observation = []
        self.a = action
        ans1 = 0
        ans2 = 0
        for i in range(0, 50):
            c = self.correr(0, self.a[0])
            if c == 0:
                ans1 += 1/100
        for i in range(0, 50):
            c = self.correr(1, self.a[0])
            if c == 0:
                ans2 += 1/100
        observation = []
        self.reward = ans1 - ans2
        if self.steps == 0:
            self.done = True
        self.steps -= 1
        return observation, self.reward, self.done

observation = []
reward = 0
theta = []
done = False
states = [-1, 1]
env = Enviroment(states)
bot = AI(0.008, [0.6], states, 0)  # alpha, theta, states
bot.eta = np.abs(bot.states[0] - bot.states[1])/2
for j in range(0, 10):
    bot.alpha = 0.008
    bot.eta = np.abs(bot.states[0] - bot.states[1])/2
    env.states = [-0.1 - j/10, 0.1 + j/10]
    bot.states = env.states
    print(f"The state {np.round(env.states[1], 2)} is starting...")
    for i in range(0, 1000):
        observation, done = env.reset()
        actions = []
        bot.rewards = []
        temporary = []
        while not done:
            action = bot.Policy(observation)
            observation, reward, done = env.step(action)
            actions.append(action[0])
            temporary.append(reward)
        theta = bot.Deep_Update(temporary, actions)

        #print(str(action) + " " + str(reward) + " " + str(theta) + " " + str(bot.states))
        bot.theta = theta

        b = open(f"Singular_AI/Data/reward_no_noise_{np.round(env.states[1], 2)}.csv", "a+", newline="")
        writer = csv.writer(b)
        writer.writerow([reward, theta[0]])
    print(f"the state {np.round(env.states[1], 2)} is saved.")
        
