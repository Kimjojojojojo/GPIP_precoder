import numpy as np

def MRT(h_hat_llk):
    J = np.size(h_hat_llk, 0)
    K = np.size(h_hat_llk, 1)
    N = np.size(h_hat_llk, 2)

    f_lk = np.zeros((J, K, N, 1), dtype=complex)

    for j in range(J):
        for k in range(K):
            h_hat = h_hat_llk[j][k]
            h_H_hat = np.conjugate(h_hat_llk[j][k]).T

            f_lk[j][k] = h_hat / (h_H_hat @h_hat)
            #f_lk[j][k] = f_lk[j][k] / np.linalg.norm(f_lk[j][k])
            #print(np.conjugate(h).T @f_lk[j][k] )


    return f_lk

def ZF(h_hat_llk):
    J = np.size(h_hat_llk, 0)
    K = np.size(h_hat_llk, 1)
    N = np.size(h_hat_llk, 2)

    H = np.zeros((J, N, K), dtype = complex)
    H_H = np.zeros((J, K, N), dtype=complex)
    F = np.zeros((J, N, K), dtype = complex)
    f_lk = np.zeros((J, K, N, 1), dtype=complex)

    for j in range(J):
        for k in range(K):
            H_H[j][k] = np.conjugate(h_hat_llk[j][k]).T
        H[j] = np.conjugate(H_H[j]).T


    for j in range(J): # ZF
        F[j] = np.linalg.inv(H@H_H)@H

    for j in range(J):
        for k in range(K):
            for n in range(N):
                f_lk[j][k][n] = F[j][:, k][n]

    return f_lk