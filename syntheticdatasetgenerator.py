# Created by elex
# Generator for synthetic datasets
# Dec 21 2017

import numpy as np
import matplotlib.pyplot as plt

# generate user and item matrix
def create_matrix(rows, cols, hmean, hstd, lmean, lstd):

    highs = np.random.normal(hmean, hstd, (int(rows/2), 1))
    lows = np.random.normal(lmean, lstd, (int(rows/2), 1))

    hl = np.concatenate([highs, lows], axis=0)
    lh = np.concatenate([lows, highs], axis=0)

    matrix = np.concatenate([hl,lh],axis=1)
    return matrix

# reconstruct rating matrix
def compute_rating_matrix(P, Q):
    R = np.dot(P,Q.T)
    return R

if __name__ == '__main__':

    k=2 # number of latent factors
    n=50 # number of users
    m=100 # number of items
    hmean = 5.0 #mean of high pref vals
    hstd = 1.0 #std of high pref vals
    lmean = 0.5 #mean of low pref vals
    lstd = 0.1 #std of low pref vals

    P = create_matrix(n, k, hmean, hstd, lmean, lstd)
    Q = create_matrix(m, k, hmean, hstd, lmean, lstd)
    R = compute_rating_matrix(P,Q)

    print("user matrix")
    print(P)
    print("item matrix")
    print(Q)
    print("rating matrix")
    print(R)

    plt.imshow(R, cmap='hot', interpolation='nearest')
    plt.show()
