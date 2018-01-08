# Created by elex
# Generator for synthetic datasets
# Dec 21 2017

import numpy as np
import matplotlib.pyplot as plt


# generate user and item matrix
def create_matrix(rows, cols, hmean, hstd, lmean, lstd):

    highs = np.random.normal(hmean, hstd, (int(rows/2), int(cols / 2)))
    lows = np.random.normal(lmean, lstd, (int(rows/2), int(cols / 2)))
    hl = np.concatenate([highs, lows], axis=0)

    highs = np.random.normal(hmean, hstd, (int(rows / 2), int(cols / 2)))
    lows = np.random.normal(lmean, lstd, (int(rows / 2), int(cols / 2)))
    lh = np.concatenate([lows, highs], axis=0)

    matrix = np.concatenate([hl, lh], axis=1)
    return matrix


# reconstruct rating matrix
def compute_rating_matrix(P, Q):
    R = np.dot(P, Q.T)
    return R


# create rating matrix
def generate_rating_matrix(k=2, n=50, m=100, hmean=5.0, hstd=1.0, lmean=0.5, lstd=0.1):
    P = create_matrix(n, k, hmean, hstd, lmean, lstd)
    Q = create_matrix(m, k, hmean, hstd, lmean, lstd)
    R = compute_rating_matrix(P, Q)

    return P, Q, R


# sample random ratings from the rating matrix
def sample_ratings(R, fraction=0.1):
    n, m = R.shape
    R_sample = np.empty((n, m))
    R_sample[:] = np.nan

    sample = np.random.choice([True, False], (n, m), p=[fraction, 1.0 - fraction])

    # check that we have at least one rating for each item
    items = np.logical_not(np.any(sample, axis=0))
    if np.any(items):
        columns = np.where(items)
        for col in columns:
            sample[np.random.randint(n), col] = True

    R_sample[sample] = R[sample]
    return R_sample


if __name__ == '__main__':

    P, Q, R = generate_rating_matrix()
    R = sample_ratings(R)

    plt.imshow(R, cmap='hot', interpolation='nearest')
    plt.show()
