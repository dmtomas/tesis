from Neural_Network import Neural_Network
import numpy as np
import csv

def Reward_for_reward(r1, r2):  # Reward given to update the predicted reward.
    ans = 1.44
    for i in range(0, len(r1)):
        ans -= ((r1[i] - r2[i])**2) / len(r1)
    return ans

class AI:
    def __init__(self, alpha, gamma, theta, delta, states, epsilon, omega):
        self.alpha = alpha
        self.alpha_r = alpha * 10
        self.gamma = gamma
        self.delta = delta
        self.states = states
        self.epsilon = epsilon
        self.neural_policy = Neural_Network(theta) # How many layers it has the Neural Network and amount of nodes.
        self.neural_reward = Neural_Network(omega)
        self.neural_policy.FullConect()
        self.neural_reward.FullConect()
        self.theta = theta
        self.quantity = 75
        self.algo_quiero_guardar = 0
        self.algo_quiero_guardar_2 = 0

    def Policy(self, s):
        out = self.neural_policy.Output(s)
        prediction = self.Predicted_Reward(s + out)
        return out, prediction

    def Predicted_Reward(self, s): # Observations = [avr1, disp1, ceros1, avr2, disp2, ceros2, D, fase, action[0], action[1]]
        out = self.neural_reward.Output(s)[0]
        return out

    def Update_Reward(self, s):
        saved_rewards = [[s[i][1], s[i][0]] for i in range(len(s))]
        prev_grad = self.neural_reward.prev_grad
        actual = [0 for i in range(len(prev_grad))]
        self.algo_quiero_guardar_2 = 0
        # This is in the direction of the gradient.
        for i in range(0, self.quantity):
            delta = np.array([np.random.uniform(-0.1, 0.1) for i in range(len(prev_grad))])
            delta = delta / (4 * np.linalg.norm(delta))
            self.neural_reward.Change_values((prev_grad + delta) * self.alpha_r)
            predicted = []
            for j in range(len(saved_rewards)):
                predicted.append(self.Predicted_Reward(saved_rewards[j][1]))

            reward = Reward_for_reward(np.array([saved_rewards[j][0] for j in range(len(saved_rewards))]), predicted)
            self.neural_reward.Change_values(-(prev_grad + delta) * self.alpha_r)
            actual += reward * (prev_grad + delta)

        for j in range(len(saved_rewards)):
            predicted.append(self.Predicted_Reward(saved_rewards[j][1]))
        self.algo_quiero_guardar_2 = Reward_for_reward(np.array([saved_rewards[j][0] for j in range(len(saved_rewards))]), predicted)
        self.neural_reward.prev_grad = actual / (4 * np.linalg.norm(actual))
        self.neural_reward.Change_values(self.neural_reward.prev_grad * self.alpha_r)
        return reward

    def Update_Policy(self, states):
        prev_grad = self.neural_policy.prev_grad
        actual = [0 for i in range(len(prev_grad))]
        self.algo_quiero_guardar = 0
        for j in range(self.quantity):
            avr_rew = 0
            deltas = np.array([np.random.uniform(-0.1, 0.1) for i in range(len(prev_grad))])
            deltas = deltas / np.linalg.norm(deltas)
            self.neural_policy.Change_values((prev_grad + deltas) * self.alpha)
            # This is in the direction of the gradient.
            for i in range(0, len(states)):
                predict = self.Policy(states[i][:-2])[0]
                reward = self.Predicted_Reward(np.array(states[i][:-2] + predict))
                avr_rew += reward / len(states)
            actual += avr_rew * (prev_grad + deltas)
            self.neural_policy.Change_values(-(prev_grad + deltas) * self.alpha)
        avr_rew = 0
        for i in range(0, len(states)):
                predict = self.Policy(states[i][:-2])[0]
                reward = self.Predicted_Reward(np.array(states[i][:-2] + predict))
                avr_rew += reward / len(states)
        self.algo_quiero_guardar = avr_rew

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
        if self.alpha_r > 0.0001:
            self.alpha_r *= 0.999
        return self.theta
    
    # Falta hacer estas 2 funciones.
    def LoadAI(self, file):
        try:
            with open(file, "r") as file:
                csvreader = csv.reader(file)
        except:
            return 0

    def SaveAI(self,file):
        nodes_policy = []
        for i in range(0, len(self.neural_policy.nodes)):
                for j in range(0, len(self.neural_policy.nodes[i])):
                        for k in range(len(self.neural_policy.nodes[i][j].weight)):
                            for p in range(len(self.neural_policy.nodes[i][j].weight[k])):
                                nodes_policy.append(self.neural_policy.nodes[i][j].weight[k][p])
        nodes_rew = []
        for i in range(0, len(self.neural_reward.nodes)):
                for j in range(0, len(self.neural_reward.nodes[i])):
                        for k in range(len(self.neural_reward.nodes[i][j].weight)):
                            for p in range(len(self.neural_reward.nodes[i][j].weight[k])):
                                nodes_rew.append(self.neural_reward.nodes[i][j].weight[k][p])
        b = open(file, "a+", newline="")
        writer = csv.writer(b)
        writer.writerows([nodes_policy, nodes_rew])
        b.close()
