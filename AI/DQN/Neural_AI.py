from Neural_Network import Neural_Network
import numpy as np
import csv

def Reward_for_reward(r1, r2):  # Reward given to update the predicted reward.
    ans = 1
    for i in range(0, len(r1)):
        ans -= (r1[i] - r2[i])**2 / len(r1)
    return ans

class AI:
    def __init__(self, alpha, gamma, theta, delta, states, epsilon, omega):
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.states = states  # We will work only on the real plane.
        self.epsilon = epsilon
        self.steps = 10000
        self.rewards = []
        self.neural_policy = Neural_Network(theta) # How many layers it has the Neural Network and amount of nodes.
        self.neural_reward = Neural_Network(omega)
        self.neural_policy.FullConect()
        self.neural_reward.FullConect()
        self.theta = theta
        self.saved_rewards = []
        self.quantity = 50

    def Policy(self, s):
        out = self.neural_policy.Output(s)
        return out

    def Predicted_Reward(self, s): # Observations = [avr1, disp1, ceros1, avr2, disp2, ceros2, D, fase, action[0], action[1]]
        a = [s[-2], s[-1]]
        disp = complex(np.cos(a[1]) * a[0], np.sin(a[1]) * a[0])
        prediction = -np.e**(-np.abs(self.states[1] + disp)**2) + np.e**(-np.abs(self.states[0] + disp)**2)
        out = self.neural_reward.Output(s)[0]
        prediction += out
        return prediction

    def Update_Reward(self, s):
        self.saved_rewards = [[s[i][1], s[i][0]] for i in range(len(s))]
        prev_grad = self.neural_reward.prev_grad
        actual = [0 for i in range(len(prev_grad))]

        # This is in the direction of the gradient.
        for i in range(0, self.quantity):
            delta = np.array([np.random.normal(-0.1, 0.1) for i in range(len(prev_grad))])
            delta = delta / (10 * np.linalg.norm(delta))
            self.neural_reward.Change_values((prev_grad + delta) * self.alpha)
            predicted = []
            for j in range(len(self.saved_rewards)):
                predicted.append(self.Predicted_Reward(self.saved_rewards[j][1]))  # Esto está mal.

            reward = Reward_for_reward(np.array([self.saved_rewards[j][0] for j in range(len(self.saved_rewards))]), predicted)
            self.neural_reward.Change_values(-(prev_grad + delta) * self.alpha)
            actual += reward * (prev_grad + delta)

        self.neural_reward.prev_grad = actual / np.linalg.norm(actual)
        self.neural_reward.Change_values(self.neural_reward.prev_grad * self.alpha)
        return reward

    def Update_Policy(self, states):
        prev_grad = self.neural_policy.prev_grad
        actual = [0 for i in range(len(prev_grad))]
        javi = []
        for j in range(self.quantity):
            avr_rew = 0
            deltas = np.array([np.random.uniform(-0.1, 0.1) for i in range(len(prev_grad))])
            deltas = deltas / (2 * np.linalg.norm(deltas))
            self.neural_policy.Change_values((prev_grad + deltas) * self.alpha)
            # This is in the direction of the gradient.
            for i in range(0, len(states)):
                predict = self.Policy(states[i][:-2])
                reward = self.Predicted_Reward(np.array(states[i][:-2] + predict)) # Acá está el bugg
                avr_rew += reward / len(states)
            actual += avr_rew * (prev_grad + deltas)
            self.neural_policy.Change_values(-(prev_grad + deltas) * self.alpha)
        #print(actual)
        self.neural_policy.prev_grad = actual / np.linalg.norm(actual)
        self.neural_policy.Change_values(self.neural_policy.prev_grad * self.alpha)

        return 0

    def Deep_Update(self, s):
        np.random.shuffle(s)
        states_reward = []
        states_policy = []
        for i in range(len(s) // 2):
            states_reward.append(s[i])
            states_policy.append(s[len(s) - 1 - i][0])
        self.Update_Policy(states_policy)
        self.Update_Reward([[states_reward[i][0], states_reward[i][1]] for i in range(len(states_reward))])
        if self.alpha > 0.0001:
            self.alpha *= 0.9999
        #self.epsilon *= 0.995
        return self.theta
    
    # Falta hacer estas 2 funciones.
    def LoadAI(self, file):
        try:
            with open(file, "r") as file:
                csvreader = csv.reader(file)
        except:
            return 0

    def SaveAI(self,file):
        with open(file, "a+") as file:
            values = csv.writer(file, delimiter=',')
            values.writerow(["Los valores de la IA"])