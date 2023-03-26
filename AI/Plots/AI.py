import numpy as np

class AI:
    def __init__(self, alpha, gamma, theta, delta, states, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta  # If we move near the 0, we can do a Taylor Expansión of a general function.
        self.delta = delta
        self.states = states  # We will work only on the real plane.
        self.epsilon = epsilon
        self.steps = 10000
        self.rewards = []

    def Policy(self, s):
        theta = [np.random.normal(self.theta[0], self.epsilon), np.random.normal(self.theta[1], self.epsilon), np.random.normal(self.theta[2], self.epsilon), np.random.normal(self.theta[2], self.epsilon)]
        disp = theta[0] + theta[1] * np.abs(self.states[1]) + theta[2] * np.abs(self.states[1])**2 + theta[3] * np.abs(self.states[1])**3
        angle = np.arctan(self.states[0].imag / self.states[0].real)
        return [disp, angle], theta


    def Gradient_Policy(self, theta):
        gradient = np.array([self.theta[0]-theta[0], self.theta[1]-theta[1], self.theta[2]-theta[2], self.theta[3]-theta[3]])
        return -gradient / np.linalg.norm(gradient)

    def Predicted_Reward(self, s): # Probability of success.
        disp = complex(np.cos(s[-1]) * s[-2], np.sin(s[-1]) * s[-2])
        return -np.e**(-np.abs(self.states[1] + disp)**2) + np.e**(-np.abs(self.states[0] + disp)**2)

    def Predicted_future(self, s, a):
        s1 = []
        disp = complex(np.cos(a[1]) * a[0], np.sin(a[1]) * a[0])
        s1 = [np.abs(disp + self.states[0])] * 2 + [np.abs(disp + self.states[1])] * 2
        return s1 + a

    def ExpectedReward(self, s, r):  # In this case we don't need to change the expected reward.
        ans = 0
        steps = 3
        ans += r
        disp = complex(np.cos(s[-1]) * s[-2], np.sin(s[-1]) * s[-2])
        for i in range(1, steps):
            ans += -np.e**(-np.abs(self.states[1] + disp)**2) + np.e**(-np.abs(self.states[0] + disp)**2) * self.gamma**steps
        return ans

    def Deep_Update(self, s, theta):
        ans = [self.theta[0], self.theta[1], self.theta[2], self.theta[3]]
        for i in range(0, len(self.theta)):
            for j in range(0, len(s)):
                ans[i] += self.ExpectedReward(s[j], self.rewards[j]) * self.Gradient_Policy(theta[j])[i] * np.e**(-(self.theta[0]-theta[j][0])**2/(2*self.epsilon**2)) * np.e**(-(self.theta[1]-theta[j][1])**2/(2*self.epsilon**2))* np.e**(-(self.theta[2]-theta[j][2])**2/(2*self.epsilon**2))* np.e**(-(self.theta[3]-theta[j][3])**2/(2*self.epsilon**2))
        ans = np.array(ans/np.linalg.norm(ans))
        self.theta += self.alpha * (ans-self.theta)  # Creo que no lo podía escribir más feo.
        self.alpha *= 0.999
        self.epsilon *= 0.999
        return self.theta