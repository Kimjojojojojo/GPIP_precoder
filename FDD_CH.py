import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

pi = np.pi

def FDD_CH_estimatior(K, L, N, lambda_ul, lambda_dl, d):
    ratio = lambda_ul / lambda_dl
    theta = np.random.uniform(0, 2*pi, (K, L))
    phase_ul = np.random.uniform(0, 2*pi, (K, L))
    phase_dl = np.random.uniform(0, 2*pi, (K, L))
    A_ul = np.zeros((K, N, L),dtype=complex)
    A_dl = np.zeros((K, N, L),dtype=complex)

    for k in range(K):
        for n in range(N):
            for l in range(L):
                A_ul[k][n][l] = np.exp(-1j * 2 * pi * (1/lambda_ul) * n * d * np.sin(theta[k][l]))

    for k in range(K):
        for n in range(N):
            for l in range(L):
                A_dl[k][n][l] = np.exp(-1j * 2 * pi * (1/lambda_dl) * n * d * np.sin(theta[k][l]))

    ### calculate eta ###
    if lambda_ul == lambda_dl:
        eta = 1
    else:
        eta = 1 / (2 * pi *(ratio - 1)) * (np.sin(2*pi*ratio) - 2j * (np.sin(pi*ratio))**2)

    ### create  g and sigma###
    B = np.zeros((K, L, 1))
    Phase_ul = np.zeros((K, L, 1),dtype=complex)
    Phase_dl = np.zeros((K, L, 1),dtype=complex) # true downlink channel phase
    Sigma = np.zeros((K, L, L))
    g_ul = np.zeros((K, L, 1),dtype=complex)
    g_dl = np.zeros((K, L, 1),dtype=complex)
    for k in range(K):
        for l in range(L):
            B[k][l][0] = np.random.uniform(0.5, 1)
            Phase_ul[k][l][0] = np.exp(1j * phase_ul[k][l])
            Phase_dl[k][l][0] = np.exp(1j * ratio * phase_ul[k][l] + 1j * (ratio - 1) * np.random.uniform(0, 2*pi))

            Sigma[k][l][l] = B[k][l][0] **2
        g_ul[k] = np.multiply(B[k], Phase_ul[k])
        g_dl[k] = np.multiply(B[k], Phase_dl[k])

    ### ture downlink channel ###
    H = np.zeros((K, N, 1),dtype=complex)
    for k in range(K):
        H[k] = A_dl[k] @ g_dl[k]

    ### create transfered g^ul ###
    g_ul_transfered = np.zeros((K, L, 1),dtype=complex)
    Phase_ul_transfered = np.zeros((K, L, 1),dtype=complex)
    for k in range(K):
        for l in range(L):
            Phase_ul_transfered[k][l][0] = np.exp(1j * ratio * phase_ul[k][l])
        g_ul_transfered[k] = np.multiply(B[k], Phase_ul_transfered[k])

    ### channel estimation ###
    H_hat_MMSE = np.zeros((K, N, 1), dtype=complex)
    H_hat_L_MMSE = np.zeros((K, N, 1),dtype=complex)
    for k in range(K):
        H_hat_MMSE[k] = eta * A_dl[k] @ g_ul_transfered[k]
        H_hat_L_MMSE[k] = np.real(eta) * A_dl[k] @ g_ul[k]

    MSE = 1 - abs(eta)**2
    L_MSE = 1 - np.real(eta)**2
    # print('MSE=',MSE)
    # print('L_MSE=',L_MSE)
    return H, H_hat_MMSE, H_hat_L_MMSE, MSE, L_MSE