import numpy as np

def Activation(out):
    return 1.2 / (1 + np.e**(-out))

class Node:
    def __init__(self):
        self.value = 0
        self.active = True
        self.conection = []
        self.weight = []

class Neural_Network:
    def __init__(self, NodesPerLayer):
        self.nodes = []
        self.prev_grad = []

        for i in range(len(NodesPerLayer)):
            self.nodes.append([Node() for j in range(0, NodesPerLayer[i])])
        

    def FullConect(self):
        for i in range(0, len(self.nodes)-1):
            for j in range(0, len(self.nodes[i])):
                for k in range(0, len(self.nodes[i+1])):
                    self.nodes[i][j].conection.append(self.nodes[i][k])
                for k in range(0, len(self.nodes[i][j].conection)):
                    self.nodes[i][j].weight.append([np.random.normal(-1, 0.1), np.random.normal(0.4, 0.1)])
                    self.prev_grad.append(0)
                    self.prev_grad.append(0)
        self.prev_grad = np.array(self.prev_grad)
        if np.linalg.norm(np.array(self.prev_grad)) != 0:
            self.prev_grad = np.array(self.prev_grad) / np.linalg.norm(np.array(self.prev_grad))
        return 0
    
    def Output(self, observation):
        for i in range(0, len(observation)):
            self.nodes[0][i].value = observation[i]
        for i in range(1, len(self.nodes)):
            for j in range(0, len(self.nodes[i])):
                value = 0
                for k in range(0, len(self.nodes[i-1])):
                    if self.nodes[i-1][k].active:
                        value += self.nodes[i-1][k].value * self.nodes[i-1][k].weight[j][0] + self.nodes[i-1][k].weight[j][1]
                self.nodes[i][j].value = Activation(value)
        out = []
        for i in range(len(self.nodes[-1])):
            out.append(self.nodes[-1][i].value)
        if len(out) == 2:
            out[-1] = 0
        return out
        
    def Change_values(self, delta):
        a = -1
        for i in range(0, len(self.nodes)):
            for j in range(0, len(self.nodes[i])):
                for k in range(0, len(self.nodes[i][j].weight)):
                    for p in range(0, len(self.nodes[i][j].weight[k])):
                        a+= 1
                        self.nodes[i][j].weight[k][p] += delta[a]
        return 0