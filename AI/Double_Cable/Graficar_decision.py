from States_Evolutions import *
import pandas as pd
import matplotlib.pyplot as plt

plots = np.array(pd.read_csv("Double_Cable/Doble_Compara_Helstrom.csv"))
decision = []
mejora = []
b = []
for i in range(len(plots)):
    b.append(plots[i][2:])

for i in range(len(b)):
    t = plots[i][0]
    covariance_1, r_1 = circ_2(-t, b[i])
    covariance_2, r_2 = circ_2(t, b[i])
    s = ""
    p_1 = np.longdouble(Calculate_Probabilities(covariance_1, r_1)/2)
    p_2 = np.longdouble(Calculate_Probabilities(covariance_2, r_2)/2)
    for i in range(0, len(p_1)):
        if p_1[i] > p_2[i]:
            s += "1"
        else:
            s += "0"
    ans = 0
    for i in range(0, len(s)):
        if s[i] == "1":
            ans += 2**i
    decision.append(ans)
    mejora.append(p_1[0]- p_2[0])
    #print(p_1[2])
plt.plot([plots[i][0] for i in range(len(plots))], decision)
#plt.plot([plots[i][0] for i in range(len(plots))], mejora)
plt.show()

