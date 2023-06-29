import numpy as np
import scipy.optimize as sp
import csv
import multiprocessing as mp

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
    exponential = np.longdouble(np.dot(np.dot(np.transpose(r), inverse), r))
    proy = np.longdouble(np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total)))
    return proy

def Vacuum_projection(A, B, C, r_a, r_b):
    A_f = A - np.longdouble(np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), np.transpose(C)))
    r_f = r_a - np.longdouble(np.dot(np.dot(C, np.linalg.inv(B + 0.5 * np.identity(2))), r_b))
    return A_f, r_f

def Single_Vacuum_probability(covariance, r):
    total = np.longdouble(covariance + 0.5 * np.identity(2))
    inverse = np.longdouble(np.linalg.inv(total))
    exponential = np.longdouble(np.dot(np.dot(np.transpose(r), inverse), r))
    proy = np.longdouble(np.e**(-0.5 * exponential) / np.sqrt(np.linalg.det(total)))
    return proy

def Calculate_Probabilities(covariance_1, r_1):
    A = np.longdouble(np.array([[covariance_1[0][0], covariance_1[0][1]], [covariance_1[1][0], covariance_1[1][1]]]))
    B = np.longdouble(np.array([[covariance_1[2][2], covariance_1[2][3]], [covariance_1[3][2], covariance_1[3][3]]]))
    C = np.array([[covariance_1[0][2], covariance_1[0][3]], [covariance_1[1][2], covariance_1[1][3]]])
    r_a = np.longdouble(np.array([r_1[0], r_1[1]]))
    r_b = np.longdouble(np.array([r_1[2], r_1[3]]))
    p = np.longdouble(np.array([0.0, 0.0, 0.0, 0.0]))  # 00, 10, 01, 11.
    p[0] = np.longdouble(Proyect_double_vacuum(covariance_1, r_1))

    A_f, r_f = Vacuum_projection(A, B, C, r_a, r_b)
    p[1] = np.longdouble((1 - Single_Vacuum_probability(A_f, r_f)) * Single_Vacuum_probability(B, r_b))

    A_f, r_f = Vacuum_projection(B, A, C, r_b, r_a)
    p[2] = np.longdouble(Single_Vacuum_probability(A, r_a) * (1 - Single_Vacuum_probability(A_f, r_f)))

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
    return ans

def Succ_prov(proy_1, proy_2):
    return np.longdouble(0.5 * proy_1 + 0.5 * (1 - proy_2))

def all_together_now(b, t):  # b == theta, z1, z2, eta, beta, gamma
    covariance_1, r_1 = circ_2(-t, b)
    covariance_2, r_2 = circ_2(t, b)
    if b[2] < b[1]:
        return 1
    return 1 - np.longdouble(General_proy(covariance_1, covariance_2, r_1, r_2))

def Optimo(data):
    a = sp.dual_annealing(all_together_now, bounds=data[0], x0=data[4], maxiter=data[1], args=(data[2],))
    data[3].send([a.fun, list(a.x)])
    data[3].close()

if __name__ == '__main__':
    bounds = [(0, 2 * np.pi), (0, 2), (0, 2), (0, 1), (0, 5), (-5, 0)]
    ans = 0
    Prev_Val = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))])
    N = 0
    times = 1
    start = 1.4725037777777779
    for i in range(0, N+1):
        if N != 0:
            t = (1.5 - start) * i / N + start
        else:
            t = start
        if t < 0.5:
            times = 50
            itera = 1000
        elif t < 1.1:
            itera = 5000
            times = 12
        else:
            itera = 15000
            times = 1
        ans = 2
        for j in range(0, times):
            caller, worker = mp.Pipe()
            cambio = Prev_Val + np.array([np.random.normal(0, 0.1) for i in range(0, 6)])
            if cambio[3] > 1:
                cambio[3] = 1
            results = []
            data = [bounds, itera, t, worker, Prev_Val]
            p1 = mp.Process(target=Optimo, args=(data,))
            p2 = mp.Process(target=Optimo, args=(data,))
            p3 = mp.Process(target=Optimo, args=(data,))
            p4 = mp.Process(target=Optimo, args=(data,))
            p5 = mp.Process(target=Optimo, args=(data,))
            p6 = mp.Process(target=Optimo, args=(data,))
            p7 = mp.Process(target=Optimo, args=(data,))
            p8 = mp.Process(target=Optimo, args=(data,))
            p9 = mp.Process(target=Optimo, args=(data,))
            p10 = mp.Process(target=Optimo, args=(data,))
            p11 = mp.Process(target=Optimo, args=(data,))
            p12 = mp.Process(target=Optimo, args=(data,))
            p1.start()
            p2.start()
            p3.start()
            p4.start()
            p5.start()
            p6.start()
            p7.start()
            p8.start()
            p9.start()
            p10.start()
            p11.start()
            p12.start()
            p1.join()
            results.append(caller.recv())
            p2.join()
            results.append(caller.recv())
            p3.join()
            results.append(caller.recv())
            p4.join()
            results.append(caller.recv())
            p5.join()
            results.append(caller.recv())
            p6.join()
            results.append(caller.recv())
            p7.join()
            results.append(caller.recv())
            p8.join()
            results.append(caller.recv())
            p9.join()
            results.append(caller.recv())
            p10.join()
            results.append(caller.recv())
            p11.join()
            results.append(caller.recv())
            p12.join()
            results.append(caller.recv())

            val = np.Infinity
            parameters = []

            for i in range(0, len(results)):
                if results[i][0] < val:
                    val = results[i][0]
                    parameters = results[i][1]
            if val < ans:
                Prev_Val = parameters
                ans = val
        print(Prev_Val)
        with open("Double_Cable/Doble_Compara_Helstrom.csv", "a+", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow([t, 1 - ans] + list(Prev_Val))