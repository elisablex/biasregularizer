# Created by elex
# Implementation of matrix factorization
# Nov 21 2017

import numpy as np
import math
import matplotlib.pyplot as plt


# Matrix factorization with Stochastic Gradient Descent
# Input:
#   R ... rating matrix
#   P ... represents strength of association between user and features, |U| x K
#   Q ... represents strength of association between items and feautures |I| x K
#   K ... number of latent features
#   s ... matrix with group membership of items
#   steps ... max number of steps to perform optimization
#   alpha ... constant, learning rate (rate of approaching the minimum with gradient descent)
#   beta ... regularization parameter, controls magnitude of user and item features so that P and Q
#           are good approximations of R without containing large numbers
#   gamma ... regularization parameter, controls impact of cut size regularization
# Output:
#   P and Q
def matrix_factorization(R, P, Q, s, steps, alpha, beta, gamma, tol=1e-5):

    m, k = Q.shape

    # mask the nans
    masked_R = np.ma.array(R, mask=np.isnan(R))

    # keep the values of the objective function
    objectives =[]

    # keep the rmse
    rmses = []

    # keep the cut size
    rs = []

    for step in range(steps):

        # calculate the gradients
        delta_P = 2.0 * np.ma.dot(np.ma.dot(P, Q.T) - masked_R, Q)
        delta_Q = 2.0 * np.ma.dot((np.ma.dot(P, Q.T) - masked_R).T, P)

        # L2 regularization
        delta_P += 2.0 * beta * P
        delta_Q += 2.0 * beta * Q

        # cut size regularization
        delta_Q += 2.0 * gamma * np.dot(np.outer(s, s), Q)
        delta_Q -= gamma * (np.tile(np.dot(Q, np.ones(k)), (k, 1)).T + np.tile(np.dot(Q.T, np.ones(m)), (m, 1)))

        # update
        P -= alpha * delta_P
        Q -= alpha * delta_Q

        # check the convergence
        objective, rmse, cs = evaluate_objective_function(masked_R, P, Q, s, beta, gamma)
        objectives.append(objective)
        rmses.append(rmse)
        rs.append(cs)
        if len(objectives) > 1:
            if abs(objectives[-1] - objectives[-2]) < tol:
                break

    # for step in range(steps):
    #     for i in range(len(R)):
    #         for j in range(len(R[i])):
    #             if R[i][j] > 0:
    #                 # error between predicted rating and real rating
    #                 eij = R[i][j] - np.dot(P[i,:],Q.T[:,j])
    #
    #                 # cut size should be maximal
    #                 # S.T * L * S
    #                 #sTrans = (np.dot(S, S.T))
    #                 #cutSize = np.dot(R, sTrans)
    #                 #cutsizeregularizer = -2 * np.dot(cutSize, Q)
    #                 for k in range(K):
    #                     #update rules for P & Q with regularization
    #                     P[i][k] = P[i][k] + alpha * (2 * eij * Q.T[k][j] - beta * (P[i][k]))# + cutsizeregularizer[i][k]))
    #                     Q.T[k][j] = Q.T[k][j] + alpha * (2 * eij * P[i][k] - beta * (Q.T[k][j]))# + cutsizeregularizer[i][k]))
    #
    #     #compute overall error to check when to end
    #     errorMF=0
    #     for i in range(len(R)):
    #         for j in range(len(R[i])):
    #             if R[i][j] > 0:
    #                 # MF error should be minimal
    #                 errorMF= errorMF + pow(R[i][j] - np.dot(P[i,:], Q.T[:,j]), 2)
    #                 for k in range(K):
    #                     errorMF = errorMF + (beta/2) * pow(P[i][k],2) + pow(Q.T[k][j],2)
    #
    #     #for now
    #     if errorMF < 0.001:
    #         break

    return P, Q, objectives, rmses, rs


# calcualte the value of the objective function
def evaluate_objective_function(masked_R, P, Q, s, beta, gamma):
    error = np.ma.dot(P, Q.T) - masked_R
    frobenius = np.linalg.norm(error)
    squared_error = frobenius * frobenius
    rmse = math.sqrt(squared_error / masked_R.count())
    cs = cut_size(graph_laplacian(adjacency(Q)), s)
    objective = squared_error + beta * (np.linalg.norm(P) ** 2 + np.linalg.norm(Q) ** 2) + \
                gamma * cs
    return objective, rmse, cs


# bisection cut size
def cut_size(L, s):
    R = np.dot(s, np.dot(L, s)) / 4
    return R


# graph laplacian
def graph_laplacian(A):
    k = np.sum(A, axis=0)
    L = np.diag(k) - A
    return L


# item adjacency matrix
def adjacency(Q):
    return np.dot(Q, Q.T)


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


def degree_matrix(Q):
    #Generate degree matrix
    A = adjacency(Q)
    vec = np.ones(len(A))

    x = np.dot(A,vec.T)
    print(x)


if __name__ == '__main__':

    # figure counter
    fc = fcounter()

    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    n, m = R.shape
    k = 3
    steps = 5000
    alpha = 0.0005  # learning rate
    beta = 0.05  # L2 regularization
    gamma = 0.05  # cut size regularization

    # division vector
    s = np.array([1, -1, -1, 1])

    # initialize P, Q with some random values
    P = np.random.rand(n, k)
    Q = np.random.rand(m, k)

    nP, nQ, objective, rmse, rs = matrix_factorization(R, P, Q, s, steps, alpha, beta, gamma)
    nR = np.dot(nP, nQ.T)

    print("approximated rating matrix")
    print(nR)

    print('adjacency')
    print(adjacency(Q))

    print('laplacian')
    print(graph_laplacian(adjacency(Q)))

    # plot the objective function and the rmse
    plot(objective, 'J', 'Objective Function')
    plot(rmse, 'RMSE', 'RMSE')
    plot(rs, 'R', 'Cut Size')
    plt.show()

