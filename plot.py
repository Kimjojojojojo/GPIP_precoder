import numpy as np
import matplotlib.pyplot as plt

def SNR_sweep(SNR_range, function_values):
    # 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_range, function_values, label = 'GPIP (with cov.)')
    plt.title('Ergodic Sum-Spectral Efficiency under imperfect CSIT')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Ergodic Sum-Spectral Efficiency [bps/Hz]')
    plt.ylim(0,10)
    plt.legend()
    plt.grid(True)
    plt.show()
