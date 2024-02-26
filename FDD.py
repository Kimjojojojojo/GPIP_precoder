import numpy as np
import FDD_CH as FDD
import matplotlib.pyplot as plt

pi = np.pi
K = 16 # number of users
L = 3 # number of multi-path
N = 16 # number of antennas

M = 10 # sample number

C = 0.3 # freq. of light [GHz]

n_f = 5
# downlink frequency
f_dl = np.array([7.125, 10, 14.3, 17.7, 21.2])
lambda_ul = C / f_dl

index = np.arange(0, 20, 1)
Band_range= np.zeros((n_f, len(index)))
lambda_dl_range = np.zeros((n_f, len(index)))
for n in range(n_f):
    Band_range[n] = np.arange(f_dl[n], f_dl[n] + 2, 0.1)
    lambda_dl_range[n] = C / Band_range[n]

MSE_MMSE_sum = np.zeros((n_f, len(index)))
MSE_L_MMSE_sum = np.zeros((n_f, len(index)))

MSE_sum = np.zeros((n_f, len(index)))
L_MSE_sum = np.zeros((n_f, len(index)))

for m in range(M):
    print(m,'th sample')
    for n in range(n_f):
        MSE_MMSE = np.zeros(len(index))
        MSE_L_MMSE = np.zeros(len(index))

        MSE = np.zeros(len(index))
        L_MSE = np.zeros(len(index))
        for idx in index:
            H, H_MMSE, H_L_MMSE, MSE[idx], L_MSE[idx] = FDD.FDD_CH_estimatior(K, L, N, lambda_ul[n], lambda_dl_range[n][idx], lambda_ul[n]/2)
            MSE_MMSE[idx] = sum([np.linalg.norm(H[k] - H_MMSE[k]) ** 2 for k in range(K)])
            MSE_L_MMSE[idx] = sum([np.linalg.norm(H[k] - H_L_MMSE[k]) ** 2 for k in range(K)])

        MSE_MMSE_sum[n] += MSE_MMSE
        MSE_L_MMSE_sum[n] += MSE_L_MMSE
        MSE_sum[n] += MSE
        L_MSE_sum[n] += L_MSE

MSE_MMSE_average = np.zeros((n_f, len(index)))
MSE_L_MMSE_average = np.zeros((n_f, len(index)))
MSE_average = np.zeros((n_f, len(index)))
L_MSE_average = np.zeros((n_f, len(index)))

for n in range(n_f):
    MSE_MMSE_average[n] = MSE_MMSE_sum[n] / M
    MSE_L_MMSE_average[n] = MSE_L_MMSE_sum[n] / M
    MSE_average[n] = MSE_sum[n] / M
    L_MSE_average[n] = L_MSE_sum[n] / M

# plt.plot(Band_range[0], MSE_MMSE_average[0], label ='MMSE' ,color = 'red')
# plt.plot(Band_range[0], MSE_L_MMSE_average[0], label ='L-MMSE' ,color = 'blue')
plt.plot(Band_range[0], MSE_average[0], label ='MSE' ,color = 'green')
plt.plot(Band_range[0], L_MSE_average[0], label ='L-MSE' ,color = 'yellow')
plt.title('MMSE & L-MMSE result comparisons')
plt.xlabel('DL carrier frequency[GHz]')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
