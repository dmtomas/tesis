import numpy as np

class AI:
    def __init__(self, alpha, theta, states, eta):
        self.alpha = alpha
        self.theta = theta
        self.rewards = []
        self.states = states
        self.eta = eta

    def Policy(self, s):
        disp = np.random.normal(self.theta[0], self.eta)
        angle = 0
        return [disp, angle]

    def Deep_Update(self, rew, actions):
        ans = 0
        for j in range(0, len(rew)):
            ans += rew[j] * np.e**(-(np.abs(actions[j] - self.theta[0])**2)) * (actions[j] - self.theta[0])/ np.abs(actions[j]-self.theta[0])
        self.theta[0] += self.alpha * ans/np.abs(ans)
        self.alpha *= 0.995
        self.eta *= 0.995
        return self.theta