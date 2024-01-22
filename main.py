import numpy as np

import calc

N = 4 # number of antenna
J = 3 # number of BSs
K = 3 # number of UEs in each BS

pi = np.pi
delta = pi / 6

R = calc.cov_h_perfect(K, N, delta) # cov. matrix of h

h = calc.ch_generation(R) # channel generation

#### I love you Eunyeong

print(h)

print(R)