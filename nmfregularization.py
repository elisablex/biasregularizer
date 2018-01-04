# Created by elex
# Implementation of regularized nonnegative matrix factorization. Script reuses some scikit-learn functions
# Goal: find two non-negative matrices (W, H), whose product approximates the non-negative matrix X.
# Jan 3 2018
import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# reused scikit-learn function
def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


# reused scikit-learn function
def NBS_init(X, n_components, init=None):
        n_samples, n_features = X.shape
        if init is None:
            if n_components < n_features:
                init = 'nndsvd'
            else:
                init = 'random'

        if init == 'nndsvd':
            W, H = _initialize_nmf(X, n_components)
        elif init == "random":
            rng = check_random_state(random_state='None')
            W = rng.randn(n_samples, n_components)
            np.abs(W, out=W)
            H = rng.randn(n_components, n_features)
            np.abs(H, H)
        else:
            raise ValueError(
                'Invalid init parameter: got %r instead of one of %r' %
                (init, (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random')))
        return W, H

# reused scikit-learn function
def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
                    random_state=None):
    """NNDSVD algorithm for NMF initialization.
    Computes a good initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    Parameters
    ----------
    X : array, [n_samples, n_features]
        The data matrix to be decomposed.
    n_components : array, [n_components, n_features]
        The number of components desired in the approximation.
    variant : None | 'a' | 'ar'
        The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
        None: leaves the zero entries as zero
        'a': Fills the zero entries with the average of X
        'ar': Fills the zero entries with standard normal random variates.
        Default: None
    eps: float
        Truncate all values less then this in output to zero.
    random_state : numpy.RandomState | int, optional
        The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random
    Returns
    -------
    (W, H) :
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.
    Remarks
    -------
    This implements the algorithm described in
    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")

    U, S, V = randomized_svd(X, n_components)
    # dtype modification
    W, H = np.zeros(U.shape, dtype=np.float32), np.zeros(V.shape,
                                                         dtype=np.float32)
    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = LA.norm(x_p), LA.norm(y_p)
        x_n_nrm, y_n_nrm = LA.norm(x_n), LA.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if variant == "a":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "ar":
        random_state = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

    return W, H

#graph normalized nmf
def regularized_nmf(X, S, lambd=0, n_components=None, max_iter=100):

        # mask the nans
        masked_X = np.ma.array(X, mask=np.isnan(X))

        masked_X = check_array(masked_X)
        check_non_negative(masked_X, "NMF.fit")
        n_samples, n_features = masked_X.shape

        if not n_components:
            n_components = min(n_samples, n_features)
        else:
            n_components = n_components

        W, H = NBS_init(masked_X, n_components, init='nndsvd')

        rmses = []

        # apply graph normalization
        A = np.dot(H.T, H) # adjacency matrix for laplacian
        D = np.matrix(np.diag(np.asarray(A).sum(axis=0)))  # degree matrix for laplacian

        for n_iter in range(1, max_iter + 1):
            delta_h1 = lambd * np.dot(H, A) + np.dot(W.T, X / (np.dot(W, H)))
            delta_h2 = lambd * np.dot(H, D) + np.dot(W.T, np.ones(X.shape))
            H = np.multiply(H, delta_h1 / delta_h2)


            delta_w1 = np.dot(masked_X / (np.dot(W, H)), H.T)
            delta_w2 = np.dot(np.ones(masked_X.shape), H.T)
            W = np.multiply(W, delta_w1 / delta_w2)

            error = LA.norm(X - np.dot(W, H))

            frobenius = np.linalg.norm(error)
            squared_error = frobenius * frobenius
            rmse = math.sqrt(squared_error / masked_X.tolist().count(1))

            rmses.append(rmse)

        return W, H, rmses

# figure counter
def fcounter():
    i = 1
    while True:
        yield i
        i += 1


# plot values vs. iteration steps
def plot(values, ylabel, title):
    plt.figure(next(fc))
    plt.plot(range(len(values)), values)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)


# plot a heatmap
def heatmap(values, title):
    plt.figure(next(fc))
    sns.heatmap(values, xticklabels=False, yticklabels=False)
    plt.title(title)


if __name__ == '__main__':

    fc = fcounter()

    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    n, m = R.shape

    # exact division vector
    S = [-1] * int(m / 2)
    S.extend([1] * int(m / 2))

    W, H, rmses = regularized_nmf(R, S, lambd=0, n_components=None, max_iter=100)

    print("W")
    print(W)
    print("H")
    print(H)
    print("reconstruction")
    R_reconstructed = np.dot(W, H)
    print(R_reconstructed)

    #plot matrices
    heatmap(R, 'R')
    heatmap(R_reconstructed, 'Reconstructed R')
    plt.show()
