import numpy as np
import scipy.optimize as sp
import csv

def BS(covariance, r, eta):
    BS = np.array([[np.sqrt(eta), 0, np.sqrt(1-eta), 0], [0, np.sqrt(eta), 0, np.sqrt(1-eta)], [-np.sqrt(1-eta), 0, np.sqrt(eta), 0], [0, -np.sqrt(1-eta), 0, np.sqrt(eta)]])
    r = np.dot(BS, r)
    covariance = np.dot(np.dot(BS, covariance), np.transpose(BS))
    return covariance, r

def SQ(covariance, r, theta, z1, z2):
    S = np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]])
    SQ = np.kron(np.array([[1, 0], [0, 0]]), np.cosh(z1) * np.identity(2) - np.sinh(z1) * S) + np.kron(np.array([[0, 0], [0, 1]]), np.cosh(z2) * np.identity(2) - np.sinh(z2) * S) 
    r = np.dot(SQ, r)
    covariance = np.dot(np.dot(SQ, covariance), np.transpose(SQ))
    return covariance, r

def Disp(r, beta, gamma):
    r = r + np.sqrt(2) * np.array([np.real(beta), np.imag(beta), np.real(gamma), np.imag(gamma)])
    return r

def circ_2(alpha, b):
    covariance = 0.5 * np.identity(4)
    r = np.array([0, 0, 0, 0])
    r = Disp(r, alpha, alpha)
    #covariance, r = SQ(covariance, r, b[0], b[1], b[2])
    #covariance, r = BS(covariance, r, b[3])
    r = Disp(r, b[4], b[5])
    return covariance, r

def Proyect_double_vacuum(covariance, r):
    total = covariance + 0.5 * np.identity(4)
    inverse = np.linalg.inv(total)
    exponential = np.dot(np.dot(r, inverse), r)
    proy = np.abs(np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total))) ** 2
    return proy

def Succ_prov(proy_1, proy_2):
    return 0.5 * proy_1 + 0.5 * (1 - proy_2)

def all_together_now(b):  # b == theta, z1, z2, eta, beta, gamma
    covariance_1, r_1 = circ_2(-t, b)
    covariance_2, r_2 = circ_2(t, b)
    proy_1 = Proyect_double_vacuum(covariance_1, r_1)
    proy_2 = Proyect_double_vacuum(covariance_2, r_2)
    return 1- Succ_prov(proy_1, proy_2)


bounds = [(0, 1.5), (0, 5), (0, 5), (0, 1), (0, 1.5), (0, 1.5)]
ans = 0
for i in range(0, 25):
    t = (0.5 - 0.05) * i / 25 + 0.05
    results = sp.dual_annealing(all_together_now, bounds=bounds)
    print(results.x)
    with open("Double_Cable/Optimized_double.csv", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([t, 1 - results.fun] + list(results.x))