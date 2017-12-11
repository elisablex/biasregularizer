# Created by elex
# Implementation of matrix factorization
# Nov 21 2017

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import linalg as LA


# Matrix factorization with Stochastic Gradient Descent
# Input:
#   R ... rating matrix
#   P ... represents strength of association between user and features, |U| x K
#   Q ... represents strength of association between items and feautures |I| x K
#   K ... number of latent features
#   steps ... max number of steps to perform optimization
#   alpha ... constant, learning rate (rate of approaching the minimum with gradient descent)
#   beta ... regularization parameter, controls magnitude of user and item features so that P and Q
#           are good approximations of R without containing large numbers
# Output:
#   P and Q
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta = 0.02):
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q.T[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q.T[k][j] - beta * P[i][k])
                        Q.T[k][j] = Q.T[k][j] + alpha * (2 * eij * P[i][k] - beta * Q.T[k][j])
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e= e + pow(R[i][j] - np.dot(P[i,:], Q.T[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * pow(P[i][k],2) + pow(Q.T[k][j],2)
        if e < 0.001:
            break
    return P, Q

if __name__ == '__main__':

    R = [
            [5,3,0,1],
            [4,0,0,1],
            [1,1,0,5],
            [1,0,0,4],
            [0,1,5,4],
        ]

    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 3

    # initialize P, Q with some random values
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = np.dot(nP, nQ.T)

    print("approximated rating matrix")
    print(nR)

