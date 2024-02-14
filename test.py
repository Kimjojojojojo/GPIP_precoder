import numpy as np
import matplotlib.pyplot as plt

import calc
import GPIP
import precoders
import plot

N = 10 # number of antenna
J = 1 # number of BSs
K = 10 # number of UEs in each BS

pi = np.pi
delta = pi / 6
sigma_tilde = 0.1
P = 10
num_iter = 30
epsilon = 0.1
beta = 0.1 # large-scale fading parameter

R_llk = calc.cov_h_perfect(K, N, delta) # cov. matrix of h : (K X (N X N))

PHI_llk = 0.1 * np.eye(N)

SNR_dB_range = np.arange(-5, 26, 5)
len_SNR = len(SNR_dB_range)
SNR_range =  10**(SNR_dB_range / 10)

R_tmp_GPIP_with_cov = np.zeros(len_SNR)
R_tmp_GPIP_without_cov = np.zeros(len_SNR)

R_tmp_GPIP_with_cov_sample_sum = np.zeros(len_SNR)
R_tmp_GPIP_without_cov_sample_sum = np.zeros(len_SNR)

R_tmp_ZF = np.zeros(len_SNR)
R_tmp_ZF_sample_sum = np.zeros(len_SNR)

##### sample average iteration #####
M = 1 # sample number
for m in range(M):
    print("-----",m+1,"th sample-----")
    for idx, s in enumerate(SNR_range): # SNR sweep
        print('SNR : ', SNR_dB_range[idx])
        h_llk = calc.rayleigh_ch_generation(R_llk, J) # channel generation h :(J X (K X N))
        h_llk_interference = calc.interference_ch_generation(R_llk, J, beta)
        #print(h_llk)
        e_llk = calc.error_generation(PHI_llk, J, K) # error vector generation h :(J X (K X N))

        h_hat_llk = h_llk + e_llk

        ##### ZF #####

        f_l_ZF = precoders.ZF(h_hat_llk)
        #print("f_l_ZF=",f_l_ZF)
        R_ZF = calc.spectral_efficiency_ZF(h_llk, h_llk_interference, f_l_ZF, PHI_llk, sigma_tilde, P, s)
        #print(R_ZF)
        R_sum_ZF = sum(sum(R_ZF))
        R_tmp_ZF[idx] = R_sum_ZF
        print(R_tmp_ZF[idx])

    ##### sample sum#####
    R_tmp_ZF_sample_sum += R_tmp_ZF

##### sample average #####
R_tmp_ZF_sample_average = R_tmp_ZF_sample_sum / M

##### plot #####
plt.figure(figsize=(10, 6))
plt.plot(SNR_dB_range, R_tmp_ZF_sample_average, label = 'ZF')
plt.title('Ergodic Sum-Spectral Efficiency under imperfect CSIT')
plt.xlabel('SNR (dB)')
plt.ylabel('Ergodic Sum-Spectral Efficiency [bps/Hz]')
plt.ylim(0,50)
plt.legend()
plt.grid(True)
plt.show()
