import numpy as np
import scipy.optimize as sp
import csv

def BS(covariance, r, eta):
    BS = np.longdouble(np.array([[np.sqrt(eta), 0.0, np.sqrt(1-eta), 0.0], [0.0, np.sqrt(eta), 0.0, np.sqrt(1-eta)], [-np.sqrt(1-eta), 0.0, np.sqrt(eta), 0.0], [0.0, -np.sqrt(1-eta), 0.0, np.sqrt(eta)]]))
    r = np.longdouble(np.dot(BS, r))
    covariance = np.longdouble(np.dot(np.dot(BS, covariance), np.transpose(BS)))
    return covariance, r

def SQ(covariance, r, theta, z1, z2):
    S = np.longdouble(np.array([[np.cos(theta), np.sin(theta)], [np.sin(theta), -np.cos(theta)]]))
    SQ = np.longdouble(np.kron(np.array([[1, 0], [0, 0]]), np.cosh(z1) * np.identity(2) - np.sinh(z1) * S) + np.kron(np.array([[0, 0], [0, 1]]), np.cosh(z2) * np.identity(2) - np.sinh(z2) * S)) 
    r = np.longdouble(np.dot(SQ, r))
    covariance = np.longdouble(np.dot(np.dot(SQ, covariance), np.transpose(SQ)))
    return covariance, r

def Disp(r, beta, gamma):
    r = r + np.longdouble(np.sqrt(2) * np.array([np.real(beta), np.imag(beta), np.real(gamma), np.imag(gamma)]))
    return r

def circ_2(alpha, b):
    covariance = np.longdouble(0.5 * np.identity(4))
    r = np.longdouble(np.array([0.0, 0.0, 0.0, 0.0]))
    r = Disp(r, alpha, alpha)
    covariance, r = SQ(covariance, r, b[0], b[1], b[2])
    covariance, r = BS(covariance, r, b[3])
    r = Disp(r, b[4], b[5])
    return covariance, r

def Proyect_double_vacuum(covariance, r):
    total = np.longdouble(covariance + 0.5 * np.identity(4))
    inverse = np.longdouble(np.linalg.inv(total))
    exponential = np.longdouble(np.dot(np.dot(r, inverse), r))
    proy = np.longdouble(np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total)))
    return proy

def Vacuum_projection(A, B, C, r_a, r_b):
    A_f = A - np.longdouble(np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), np.transpose(C)))
    r_f = r_a - np.longdouble(np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), r_b))
    return A_f, r_f

def Single_Vacuum_probability(covariance, r):
    total = np.longdouble(covariance + 0.5 * np.identity(2))
    inverse = np.longdouble(np.linalg.inv(total))
    exponential = np.longdouble(np.dot(np.dot(r, inverse), r))
    proy = np.longdouble(np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total)))
    return proy

def Calculate_Probabilities(covariance_1, r_1):
    A = np.longdouble(np.array([[covariance_1[0][0], covariance_1[0][1]], [covariance_1[1][0], covariance_1[1][1]]]))
    B = np.longdouble(np.array([[covariance_1[2][2], covariance_1[2][3]], [covariance_1[3][2], covariance_1[3][3]]]))
    #C = np.array([[covariance_1[0][2], covariance_1[0][3]], [covariance_1[1][2], covariance_1[1][3]]])
    r_a = np.longdouble(np.array([r_1[0], r_1[1]]))
    r_b = np.longdouble(np.array([r_1[2], r_1[3]]))
    p = np.longdouble(np.array([0.0, 0.0, 0.0, 0.0]))  # 00, 10, 01, 11.
    p[0] = np.longdouble(Proyect_double_vacuum(covariance_1, r_1))

    p[1] = np.longdouble((1 - Single_Vacuum_probability(A, r_a)) * Single_Vacuum_probability(B, r_b))

    p[2] = np.longdouble(Single_Vacuum_probability(A, r_a) * (1 - Single_Vacuum_probability(B, r_b)))

    p[3] = np.longdouble(1 - p[0] - p[1] - p[2])
    return p

def General_proy(covariance_1, covariance_2, r_1, r_2):
    p_1 = np.longdouble(Calculate_Probabilities(covariance_1, r_1)/2)
    p_2 = np.longdouble(Calculate_Probabilities(covariance_2, r_2)/2)

    ans = np.longdouble(0.0)
    for i in range(0, len(p_1)):
        if p_1[i] > p_2[i]:
            ans += p_1[i]
        else:
            ans += p_2[i]
    if p_1[3] < 0 or p_2[3] < 0:
        ans = 0
    return ans

def Succ_prov(proy_1, proy_2):
    return np.longdouble(0.5 * proy_1 + 0.5 * (1 - proy_2))

def all_together_now(b):  # b == theta, z1, z2, eta, beta, gamma
    covariance_1, r_1 = circ_2(-t, b)
    covariance_2, r_2 = circ_2(t, b)
    if b[2] < b[1]:
        return 1
    return 1 - np.longdouble(General_proy(covariance_1, covariance_2, r_1, r_2))


bounds = [(0, 2 * np.pi), (0, 2), (0, 2), (0, 1), (0, 5), (0, 5)]
ans = 0
Prev_Val = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))])
N = 50
times = 1
for i in range(0, N+1):
    t = (1.5 - 0.05) * i / N + 0.05
    if t < 0.5:
        times = 30
        itera = 1000
    elif t < 0.8:
        itera = 5000
        times = 5
    else:
        itera = 10000
        times = 1
    ans = 2
    vals = []
    for j in range(0, times):
        cambio = Prev_Val + np.array([np.random.normal(0, 0.1) for i in range(0, 6)])
        if cambio[3] > 1:
            cambio[3] = 1
        results = sp.dual_annealing(all_together_now, bounds=bounds, x0=cambio, maxiter=itera)
        if results.fun < ans:
            Prev_Val = results.x
            ans = results.fun
    print(Prev_Val)
    with open("Double_Cable/Doble_Compara_Helstrom.csv", "a+", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([t, 1 - ans] + list(Prev_Val))