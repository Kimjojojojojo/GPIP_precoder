from scipy import integrate
import numpy as np

def integrand_re(x, n):
    pi = np.pi
    return np.cos(pi * np.sqrt(n) / np.sqrt(2) * (np.cos(x) + np.sin(x)))

def integrand_im(x, n):
    pi = np.pi
    return np.sin(pi * np.sqrt(n) / np.sqrt(2) * (np.cos(x) + np.sin(x)))

def cov_h_perfect(K,N, delta):
    pi = np.pi

    R = np.zeros((K,N,N)) # cov. matrix of channel
    for k in range(K):
        for i in range(N):
            for j in range(N):
                n = abs(i-j)
                upper = 2 * pi * k / K + delta
                lower = 2 * pi * k / K - delta

                # quad 함수를 사용하여 적분 계산
                result1, error1 = integrate.quad(lambda x: integrand_re(x, n), lower, upper)
                result2, error2 = integrate.quad(lambda x: integrand_im(x, n), lower, upper)

                result = abs(result1 - 1j*result2)
                R[k][i][j] = result / (2*delta)
    return R


def ch_generation(R):
    K = np.size(R,0)
    N = np.size(R, 1)

    h = np.zeros((K, N))
    for k in range(K):
        mean = np.zeros(N)
        h[k] = np.random.multivariate_normal(mean, R[k], 1)

    return h