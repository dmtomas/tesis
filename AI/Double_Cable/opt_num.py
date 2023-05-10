import numpy as np
import scipy.optimize as sp
import csv

def doble_opt(b):  # This is the error function that is going to be minimized.
    return 1 - (0.5 * np.e**(-2 * (t-b[0])**2) + 0.5 * (1 - np.e**(-2 *(t + b[0])**2)))

bounds = [(0.5, 1.5)]
ans = 0
for i in range(1, 101):
    t = 0.5 * i / 100
    results = sp.dual_annealing(doble_opt, bounds=bounds)
    #print(results)
    with open("Double_Cable/Kennedy_Opt_Num.csv", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([t, results.x[0], 1 - results.fun])