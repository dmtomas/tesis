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
    covariance, r = SQ(covariance, r, b[0], b[1], b[2])
    covariance, r = BS(covariance, r, b[3])
    r = Disp(r, b[4], b[5])
    return covariance, r

def Proyect_double_vacuum(covariance, r):
    total = covariance + 0.5 * np.identity(4)
    inverse = np.linalg.inv(total)
    exponential = np.dot(np.dot(r, inverse), r)
    proy = np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total))
    return proy

def Vacuum_projection(A, B, C, r_a, r_b):
    A_f = A - np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), np.transpose(C))
    r_f = r_a - np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), r_b)
    return A_f, r_f

def Single_Vacuum_probability(covariance, r):
    total = covariance + 0.5 * np.identity(2)
    inverse = np.linalg.inv(total)
    exponential = np.dot(np.dot(r, inverse), r)
    proy = np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total))
    return proy

def Calculate_Probabilities(covariance_1, r_1):
    A = np.array([[covariance_1[0][0], covariance_1[0][1]], [covariance_1[1][0], covariance_1[1][1]]])
    B = np.array([[covariance_1[2][2], covariance_1[2][3]], [covariance_1[3][2], covariance_1[3][3]]])
    C = np.array([[covariance_1[0][2], covariance_1[0][3]], [covariance_1[1][2], covariance_1[1][3]]])
    r_a = np.array([r_1[0], r_1[1]])
    r_b = np.array([r_1[2], r_1[3]])
    p = np.array([0.0, 0.0, 0.0, 0.0])  # 00, 10, 01, 11.
    p[0] = Proyect_double_vacuum(covariance_1, r_1)

    A_f, r_f = Vacuum_projection(A, B, C, r_a, r_b)
    p[1] = 1 - Single_Vacuum_probability(A_f, r_f)

    A_f, r_f = Vacuum_projection(B, A, C, r_b, r_a)
    p[2] = 1 - Single_Vacuum_probability(A_f, r_f)

    p[3] = 1 - p[0] - p[1] - p[2]
    return p

def General_proy(covariance_1, covariance_2, r_1, r_2):
    p_1 = Calculate_Probabilities(covariance_1, r_1)/2
    p_2 = Calculate_Probabilities(covariance_2, r_2)/2
    print(str(p_1) + " " + str(p_2))

    ans = 0
    for i in range(0, len(p_1)):
        for j in range(i + 1, len(p_1)):
            otros = []
            for k in range(len(p_1)):
                if k != i and k != j:
                    otros.append(k)
            if ans < p_1[i] + p_1[j] + p_2[otros[0]] + p_2[otros[1]]:
                ans = p_1[i] + p_1[j] + p_2[otros[0]] + p_2[otros[1]]
    return ans

def Succ_prov(proy_1, proy_2):
    return 0.5 * proy_1 + 0.5 * (1 - proy_2)

def all_together_now(b):  # b == theta, z1, z2, eta, beta, gamma
    covariance_1, r_1 = circ_2(-t, b)
    covariance_2, r_2 = circ_2(t, b)
    return 1 - General_proy(covariance_1, covariance_2, r_1, r_2)


bounds = [(0, 2 * np.pi), (0, 5), (0, 5), (0, 1), (0, 1.5), (0, 1.5)]
ans = 0
for i in range(1, 11):
    t = 0.5 * i / 10
    results = sp.dual_annealing(all_together_now, bounds=bounds)
    print(results.x)
    with open("Double_Cable/Optimized_double.csv", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([t, 1 - results.fun] + list(results.x))