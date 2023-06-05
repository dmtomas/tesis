import numpy as np
import scipy.optimize as sp
import csv

def doble_opt(b):  # This is the error function that is going to be minimized.
    return 0.5 * np.e**(-2 * ((t + b[0])**2)) + 0.5 * (1- np.e**(-2 * ((-t + b[0])**2)))

bounds = [(0, 1.5)]
ans = 0
N = 50

for i in range(0, N):
    t = (1.5-0.05) * i / N + 0.05
    results = sp.dual_annealing(doble_opt, bounds=bounds)
    print(results.x)
    with open("Double_Cable/Kennedy_Opt_Num.csv", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([t, results.x[0], 1 - results.fun])