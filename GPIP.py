import numpy as np

def precoder_generation(h_hat_llk, PHI_llk, P, sigma_tilde, num_iter, epsilon, s): # h_hat_llk : (J X (K X N))
    J = np.size(h_hat_llk, 0)
    K = np.size(h_hat_llk, 1)
    N = np.size(h_hat_llk, 2)

    f_lk = np.zeros((J, K, N))
    for j in range(J): # f_lk initialization
        for k in range(K):
            f_lk[j][k] = np.conjugate(h_hat_llk[j][k])

    f_l = np.zeros((J, N*K, 1)) # concat. of f_llk
    f_l_tmp = np.zeros((num_iter, J, N * K, 1))

    A_llk = np.zeros((J, K, N * K, N * K))
    B_llk = np.zeros((J, K, N * K, N * K))
    # print(A_llk[0][0][0:4, 0:4])
    for j in range(J):
        for k in range(K):
            tmp = h_hat_llk[j][k].T @ np.conjugate(h_hat_llk[j][k].T).T + PHI_llk
            for n in range(K):
                A_llk[j][k][n * N: n * N + N, n * N: n * N + N] = tmp
                B_llk[j][k][n * N: n * N + N, n * N: n * N + N] = tmp
            A_llk[j][k] = A_llk[j][k] + 1/s * np.eye(N * K)
            B_llk[j][k] = B_llk[j][k] + 1/s * np.eye(N * K)
            B_llk[j][k][k * N: k * N + N, k * N: k * N + N] = np.zeros((N, N))

    for j in range(J):
        for k in range(K):
            for n in range(N):
                f_l[j][k * N + n] = f_lk[j][k][n]

    num_iterend = 0
    ### iteration start ###
    for num in range(num_iter):
        f_l_tmp[num] = f_l # save f_l
        end_critic = 0
        for j in range(J):
            end_critic += np.linalg.norm(f_l_tmp[num][j] - f_l_tmp[num-1][j])

        if num >= 3 and end_critic <= epsilon:
            num_iterend = num
            break

        A_bar_ll = np.zeros((J, N * K, N * K))
        B_bar_ll = np.zeros((J, N * K, N * K))
        B_bar_ll_inv = np.zeros((J, N * K, N * K))
        w_li = np.ones((J,K))
        for j in range(J):
            if sum(abs(f_l[j])) == 0:
                continue
            for i in range(K):
                tmpA_1 = np.conjugate(f_l[j]).T @ A_llk[j][i] @ f_l[j]
                tmpB_1 = np.conjugate(f_l[j]).T @ B_llk[j][i] @ f_l[j]
                tmpA_2 = 1
                tmpB_2 = 1
                for k in range(K):
                    if k == i :
                        continue
                    tmpA_2 = tmpA_2 * np.conjugate(f_l[j]).T @ A_llk[j][k] @ f_l[j]
                    tmpB_2 = tmpB_2 * np.conjugate(f_l[j]).T @ B_llk[j][k] @ f_l[j]
                # print('tmpB_2=',tmpB_2)
                A_bar_ll[j] = A_bar_ll[j] + w_li[j][i] * ((tmpA_1) ** (w_li[j][i] - 1)) * tmpA_2 * A_llk[j][i]
                B_bar_ll[j] = B_bar_ll[j] + w_li[j][i] * ((tmpB_1) ** (w_li[j][i] - 1)) * tmpB_2 * B_llk[j][i]
            # print('f_l[j]=', f_l[j])
            # print('B_bar=',B_bar_ll[j])
            B_bar_ll_inv[j] = np.linalg.inv(B_bar_ll[j])
        print('-------------------------------------------------')
        for j in range(J):
            #print('before =', f_l[j].T)
            # print('f_l[j]=', f_l[j])
            f_l[j] = B_bar_ll_inv[j] @ A_bar_ll[j] @ f_l[j]
            # print('updated f_l[j]=', f_l[j])
            #print('after =', f_l[j].T)
            f_l[j] = f_l[j] / np.linalg.norm(f_l[j])
            # print('normalized f_l[j]=', f_l[j])
            # print('B_bar_inv=,', B_bar_ll_inv[j])
            # print('A_bar=,', A_bar_ll[j])

    return f_l_tmp, num_iterend

