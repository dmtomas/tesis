import numpy as np
import matplotlib.pyplot as plt

def Poisson(alpha, n):
    return np.abs(alpha)**(2 * n) * np.e**(-np.abs(alpha)**2) / np.math.factorial(n)
alpha = 2

prob = [Poisson(alpha, 0)]
x = []
for i in range(0, 1000000):
    ans = np.random.uniform(0, 1)
    while ans > prob[-1]:
        prob.append(prob[-1] + Poisson(alpha, len(prob)))
    for j in range(0, len(prob)):
        if ans < prob[j]:
            x.append(j)
            break


plt.hist(x, max(x), density=True)

plt.show()
