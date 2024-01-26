import numpy as np

import calc
import GPIP
import plot

N = 2 # number of antenna
J = 2 # number of BSs
K = 2 # number of UEs in each BS

pi = np.pi
delta = pi / 6
sigma_tilde = 0.1
P = 10
num_iter = 30
epsilon = 0.1


R_llk = calc.cov_h_perfect(K, N, delta) # cov. matrix of h : (K X (N X N))

PHI_llk = 0.1 * np.eye(N)

SNR_dB_range = np.arange(-5, 26, 5)
len_SNR = len(SNR_dB_range)
SNR_range = 10 * np.log10(1 + 10**(SNR_dB_range / 10))

R_tmp = np.zeros(len_SNR)
R_tmp_sample_sum = np.zeros(len_SNR)
M = 1 # sample number
for m in range(M):
    for idx, s in enumerate(SNR_range): # SNR sweep
        print('SNR : ', SNR_dB_range[idx])
        h_llk = calc.ch_generation(R_llk, J) # channel generation h :(J X (K X N))

        e_llk = calc.error_generation(PHI_llk, J, K) # error vector generation h :(J X (K X N))

        h_hat_llk = h_llk + e_llk

        f_l_tmp, num_iterend = GPIP.precoder_generation(h_hat_llk, PHI_llk, P, sigma_tilde, num_iter ,epsilon, s)# pre-coder generation f : (J X (K X N))
        f_l = f_l_tmp[num_iterend]

        R = calc.spectral_efficiency(h_hat_llk, f_l, PHI_llk, sigma_tilde, P, s) # data rate R : (J X K)
        R_sum = sum(sum(R))
        print(R_sum)
        R_tmp[idx] = R_sum
    R_tmp_sample_sum += R_tmp

R_tmp_sample_average = R_tmp_sample_sum/M
plot.SNR_sweep(SNR_range, R_tmp_sample_average)
# print(R_sum)


