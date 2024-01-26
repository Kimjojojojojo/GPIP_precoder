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


def ch_generation(R, J):
    K = np.size(R,0)
    N = np.size(R, 1)

    x = np.zeros((J, K, N))
    mean = np.zeros(N)
    for j in range(J):
        for k in range(K):
            x[j][k] = np.random.multivariate_normal(mean, R[k], 1)

    return x

def error_generation(PHI, J, K):
    N = np.size(PHI, 0)

    x = np.zeros((J, K, N))
    mean = np.zeros(N)
    for j in range(J):
        for k in range(K):
            x[j][k] = np.random.multivariate_normal(mean, PHI, 1)

    return x

def spectral_efficiency(h_hat_llk, f_l, PHI_llk, sigma_tilde, P, s):
    J = np.size(h_hat_llk, 0)
    K = np.size(h_hat_llk, 1)
    N = np.size(h_hat_llk, 2)

    f_lk = np.zeros((J, K, N, 1))
    f_H_lk = np.zeros((J, K, 1, N))
    for j in range(J):
        for k in range(K):
            f_lk[j][k] = f_l[j][k*N:(k+1)*N]
            f_H_lk[j][k] = np.conjugate(f_lk[j][k]).T

    h_H_hat_llk = np.zeros((J, K, N))
    for j in range(J):
        for k in range(K):
            h_H_hat_llk[j][k] = np.conjugate(h_hat_llk[j][k]).T

    R = np.zeros((J,K))
    for j in range(J):
        for k in range(K):
            a = abs(h_H_hat_llk[j][k] @ f_lk[j][k])**2
            b = sum([abs(h_H_hat_llk[j][k] @ f_lk[j][i])**2 for i in range(K)]) - a
            c = sum([abs(f_H_lk[j][i] @ PHI_llk @f_lk[j][i])**2 for i in range(K)])
            d = 1/s
            R[j][k] = np.log2(1 + a/(b+ c+ d))

    return R