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

R_tmp_MRT = np.zeros(len_SNR)
R_tmp_MRT_sample_sum = np.zeros(len_SNR)

##### sample average iteration #####
M = 100 # sample number
for m in range(M):
    print("-----",m+1,"th sample-----")
    for idx, s in enumerate(SNR_range): # SNR sweep
        print('SNR : ', SNR_dB_range[idx])
        h_llk = calc.rayleigh_ch_generation(R_llk, J) # channel generation h :(J X (K X (NX1)))
        h_llk_interference = calc.interference_ch_generation(R_llk, J, beta)

        e_llk = calc.error_generation(PHI_llk, J, K) # error vector generation h :(J X (K X (NX1)))

        h_hat_llk = h_llk
        #print('--------------------------------')
        #####GPIP with covariance matrix#####

        f_l_tmp_GPIP_with_cov, num_iterend_GPIP_with_cov = GPIP.precoder_generation(h_hat_llk, PHI_llk, P, sigma_tilde,                                                                     num_iter, epsilon, s)# pre-coder generation f : (J X (K X N))
        f_l_GPIP_with_cov = f_l_tmp_GPIP_with_cov[num_iterend_GPIP_with_cov]
        #print("f_l_with_cov=",np.linalg.norm(f_l_GPIP_with_cov)**2)
        R_GPIP_with_cov = calc.spectral_efficiency(h_llk, h_llk_interference, f_l_GPIP_with_cov, PHI_llk, sigma_tilde, P, s) # data rate R : (J X K)
        R_sum_GPIP_with_cov = sum(sum(R_GPIP_with_cov))
        R_tmp_GPIP_with_cov[idx] = R_sum_GPIP_with_cov

        #####GPIP without covariance matrix#####

        f_l_tmp_GPIP_without_cov, num_iterend_GPIP_with_cov = GPIP.precoder_generation(h_hat_llk, 0*np.eye(N), P, sigma_tilde,
                                                                                    num_iter, epsilon,                                                                                s)  # pre-coder generation f : (J X (K X N))
        f_l_GPIP_without_cov = f_l_tmp_GPIP_without_cov[num_iterend_GPIP_with_cov]
        #print("f_l_without_cov=", np.linalg.norm(f_l_GPIP_without_cov)**2)
        R_GPIP_without_cov = calc.spectral_efficiency(h_llk, h_llk_interference, f_l_GPIP_without_cov, PHI_llk, sigma_tilde, P,
                                                   s)  # data rate R : (J X K)
        R_sum_GPIP_without_cov = sum(sum(R_GPIP_without_cov))
        R_tmp_GPIP_without_cov[idx] = R_sum_GPIP_without_cov

        ##### ZF #####

        f_l_ZF = precoders.ZF(h_hat_llk)
        R_ZF = calc.spectral_efficiency_ZF(h_llk, h_llk_interference, f_l_ZF, PHI_llk, sigma_tilde, P, s)
        R_sum_ZF = sum(sum(R_ZF))
        R_tmp_ZF[idx] = R_sum_ZF

        ##### MRT #####
        f_l_MRT = precoders.MRT(h_hat_llk)
        R_MRT = calc.spectral_efficiency_ZF(h_llk, h_llk_interference, f_l_MRT, PHI_llk, sigma_tilde, P, s)
        R_sum_MRT = sum(sum(R_MRT))
        R_tmp_MRT[idx] = R_sum_MRT

    ##### sample sum#####
    R_tmp_GPIP_with_cov_sample_sum += R_tmp_GPIP_with_cov
    R_tmp_GPIP_without_cov_sample_sum += R_tmp_GPIP_without_cov
    R_tmp_ZF_sample_sum += R_tmp_ZF
    R_tmp_MRT_sample_sum += R_tmp_MRT

##### sample average #####
R_tmp_with_cov_sample_average = R_tmp_GPIP_with_cov_sample_sum / M
R_tmp_without_cov_sample_average = R_tmp_GPIP_without_cov_sample_sum / M
R_tmp_ZF_sample_average = R_tmp_ZF_sample_sum / M
R_tmp_MRT_sample_average = R_tmp_MRT_sample_sum / M

##### plot #####
plt.figure(figsize=(10, 6))
#plt.plot(SNR_dB_range, R_tmp_with_cov_sample_average, color = 'red')
#plt.scatter(SNR_dB_range, R_tmp_with_cov_sample_average, label ='GPIP (with cov.)', edgecolors = 'red', marker = 'o', facecolor='none',s=100)

plt.plot(SNR_dB_range, R_tmp_without_cov_sample_average,  color = 'red')
plt.scatter(SNR_dB_range, R_tmp_without_cov_sample_average, label ='GPIP',edgecolors = 'red', marker = 'o', facecolor='none',s=100)

plt.plot(SNR_dB_range, R_tmp_ZF_sample_average, color = 'black')
plt.scatter(SNR_dB_range, R_tmp_ZF_sample_average,  label = 'ZF',edgecolors = 'black', marker= 's', facecolor='none', s=100)

plt.plot(SNR_dB_range, R_tmp_MRT_sample_average,  color = 'black')
plt.scatter(SNR_dB_range, R_tmp_MRT_sample_average, label = 'MRT',color = 'black', marker= 'x', s=100)

plt.title('Ergodic Sum-Spectral Efficiency Under Perfect CSIT')
plt.xlabel('SNR (dB)')
plt.ylabel('Ergodic Sum-Spectral Efficiency [bps/Hz]')
plt.ylim(0,60)
plt.legend()
plt.grid(True)
plt.show()
