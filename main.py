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
#   S ... matrix with group membership of items
#   steps ... max number of steps to perform optimization
#   alpha ... constant, learning rate (rate of approaching the minimum with gradient descent)
#   beta ... regularization parameter, controls magnitude of user and item features so that P and Q
#           are good approximations of R without containing large numbers
# Output:
#   P and Q
def matrix_factorization(R, P, Q, k, s, steps, eta, lam, tol=1e-5):

    # mask the nans
    masked_R = np.ma.array(R, mask=np.isnan(R))

    # keep the values of the obejctive function
    objectives =[]

    # keep the rmse
    rmses = []

    for step in range(steps):

        # calculate the gradients
        delta_P = np.ma.dot(np.ma.dot(P, Q.T) - masked_R, Q)
        delta_Q = np.ma.dot((np.ma.dot(P, Q.T) - masked_R).T, P)

        # regularize
        delta_P += lam * P
        delta_Q += lam * Q

        # update
        P -= 2.0 * eta * delta_P
        Q -= 2.0 * eta * delta_Q

        # check the convergence
        objective, rmse = evaluate_objective_function(masked_R, P, Q, lam)
        objectives.append(objective)
        rmses.append(rmse)
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

    return P, Q, objectives, rmses


# calcualte the value of the objective function
def evaluate_objective_function(masked_R, P, Q, lam):
    error = np.ma.dot(P, Q.T) - masked_R
    frobenius = np.linalg.norm(error)
    squared_error = frobenius * frobenius
    rmse = math.sqrt(squared_error / masked_R.count())
    objective = squared_error + lam * (np.linalg.norm(P) ** 2 + np.linalg.norm(Q) ** 2)
    return objective, rmse


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


def laplacian(S,Q):

    #weighted similarity matrix
    A = np.dot(Q,Q.T)
    #Generate degree matrix
    vec = np.ones(len(Q))

    x = np.dot(A,vec.T)
    print(x)
    #compute laplacian of weighted similarity matrix


if __name__ == '__main__':

    # figure counter
    fc = fcounter()

    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])

    n, m = R.shape
    k = 2
    steps = 1000
    eta = 0.001  # learning rate
    lam = 0.001  # regularization strength

    # division vector
    s = np.array([1, -1, -1, 1])

    # initialize P, Q with some random values
    P = np.random.rand(n, k)
    Q = np.random.rand(m, k)

    nP, nQ, objective, rmse = matrix_factorization(R, P, Q, k, s, steps, eta, lam)
    nR = np.dot(nP, nQ.T)

    print("approximated rating matrix")
    print(nR)

    # plot the objective function and the rmse
    plot(objective, 'J', 'Objective Function')
    plot(rmse, 'RMSE', 'RMSE')
    plt.show()

