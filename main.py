# Created by elex
# Implementation of matrix factorization
# Nov 21 2017

import numpy as np


# Matrix factorization with Stochastic Gradient Descent
# Input:
#   R ... rating matrix
#   P ... represents strength of association between user and features, |U| x K
#   Q ... represents strength of association between items and feautures |I| x K
#   K ... number of latent features
#   S ... matrix with group membership of items
#   steps ... max number of steps to perform optimization
#   alpha ... constant, learning rate (rate of approaching the minimum with gradient descent)
#   beta ... regularization parameter, controls magnitude of user and item features so that P and Q
#           are good approximations of R without containing large numbers
# Output:
#   P and Q
def matrix_factorization(R, P, Q, K, S, steps, alpha, beta):
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # error between predicted rating and real rating
                    eij = R[i][j] - np.dot(P[i,:],Q.T[:,j])
                    # cut size should be maximal
                    sTrans = (np.dot(S, S.T))
                    cutSize = np.dot(R, sTrans)
                    cutsizeregularizer = -2 * np.dot(cutSize, Q)
                    for k in range(K):
                        #update rules for P & Q with regularization
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q.T[k][j] - beta * (P[i][k] + cutsizeregularizer[i][k]))
                        Q.T[k][j] = Q.T[k][j] + alpha * (2 * eij * P[i][k] - beta * (Q.T[k][j] + cutsizeregularizer[i][k]))

        #compute overall error to check when to end
        errorMF=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # MF error should be minimal
                    errorMF= errorMF + pow(R[i][j] - np.dot(P[i,:], Q.T[:,j]), 2)
                    for k in range(K):
                        errorMF = errorMF + (beta/2) * pow(P[i][k],2) + pow(Q.T[k][j],2)

        #for now
        if errorMF < 0.001:
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
    K = 2
    steps = 500
    alpha = 0.001
    beta = 0.0001

    S = [
            [1],
            [-1],
            [-1],
            [1],
    ]
    S = np.array(S)
    # initialize P, Q with some random values
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K, S, steps, alpha, beta)
    nR = np.dot(nP, nQ.T)

    print("approximated rating matrix")
    print(nR)

