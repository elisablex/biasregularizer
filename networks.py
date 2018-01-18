import numpy as np


def katz_similarity(A):
    n, n, = A.shape
    lambd, v = np.linalg.eig(A)
    rho = np.absolute(lambd).max()
    alpha = 0.85 / rho

    paths = alpha * A
    katz = np.eye(n) + paths
    for i in range(100):
        katz += alpha * A * paths

    return katz
