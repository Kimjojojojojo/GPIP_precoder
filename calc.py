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

    x = np.zeros((J, K, N, 1))
    mean = np.zeros(N)
    for j in range(J):
        for k in range(K):
            x[j][k] = np.transpose(np.random.multivariate_normal(mean, R[k], 1))

    return x

def interference_ch_generation(R, J, beta):
    K = np.size(R,0)
    N = np.size(R, 1)

    h = np.zeros((J, J, K, N))
    mean = np.zeros(N)
    for j in range(J):
        for l in range(J):
            if j == l:
                continue
            for k in range(K):
                h[j][l][k] = np.random.multivariate_normal(mean, beta * R[k], 1)

    return h

def rayleigh_ch_generation(R, J):
    K = np.size(R,0)
    N = np.size(R, 1)

    h = np.zeros((J, K, N, 1), dtype=complex)
    mean = np.zeros(N)
    for j in range(J):
        for k in range(K):
            h[j][k] = (np.random.normal(0, 1, (N,1)) + 1j*np.random.normal(0, 1, (N,1)))/np.sqrt(2)

    return h

def error_generation(PHI, J, K):
    N = np.size(PHI, 0)

    x = np.zeros((J, K, N, 1),dtype=complex)
    mean = np.zeros(N)
    for j in range(J):
        for k in range(K):
            x[j][k] = np.transpose((np.random.multivariate_normal(mean, PHI[k], 1) + 1j*np.random.multivariate_normal(mean, PHI[k], 1))/np.sqrt(2))

    return x

def spectral_efficiency(h_llk, h_llk_interference, f_l, PHI_llk, sigma_tilde, P, s):
    J = np.size(h_llk, 0)
    K = np.size(h_llk, 1)
    N = np.size(h_llk, 2)

    f_lk = np.zeros((J, K, N, 1),dtype=complex)
    for j in range(J):
        for k in range(K):
            f_lk[j][k] = f_l[j][k*N:(k+1)*N]

    #print(f_lk[0][0])
    R = np.zeros((J,K))
    for j in range(J):
        for k in range(K):
            h = h_llk[j][k]
            h_H = np.conjugate(h).T
            #print(h_H[j][k])
            #print(f_lk[j][k])
            a = abs(h_H @ f_lk[j][k])**2
            #print('GPIPa=',a)
            b = sum([abs(h_H @ f_lk[j][i])**2 for i in range(K)]) - a
            #print('GPIPb=', b)
            # c = 0
            # for l in range(J):
            #     if l == j:
            #         continue
            #     c = sum([abs(np.conjugate(h_llk_interference[j][l][k]).T @ f_lk[j][i])**2 for i in range(K)])
            d = 1/s
            R[j][k] = np.log2(1 + a/(b + d))

    return R

def spectral_efficiency_ZF(h_llk, h_llk_interference, f_lk, PHI_llk, sigma_tilde, P, s):
    J = np.size(h_llk, 0)
    K = np.size(h_llk, 1)
    N = np.size(h_llk, 2)

    R = np.zeros((J, K))
    for j in range(J):
        for k in range(K):
            h = h_llk[j][k]
            h_H = np.conjugate(h).T
            a = abs(h_H @ f_lk[j][k]) ** 2

            #print('ZFa=',a)
            #print([abs(h_H @ f_lk[j][i]) ** 2 for i in range(K)])
            b = sum([abs(h_H @ f_lk[j][i]) ** 2 for i in range(K)]) - a
            #print('ZFb=',b)
            # c = 0
            # for l in range(J):
            #     if l == j:
            #         continue
            #     c = sum([abs(np.conjugate(h_llk_interference[j][l][k]).T @ f_lk[j][i])**2 for i in range(K)])
            d = 1 / s
            #print('ZFd=',d)
            #print('x=',1 + a / (b + d))
            R[j][k] = np.log2(1 + a / (b + d))

    return R