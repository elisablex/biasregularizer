# Created by elex
# Implementation of matrix factorization
# Nov 21 2017

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from matrixgenerator import generate_rating_matrix


# Matrix factorization with Stochastic Gradient Descent
# Input:
#   R ... rating matrix
#   P ... represents strength of association between user and features, |U| x k
#   Q ... represents strength of association between items and feautures |I| x k
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

    return P, Q, objectives, rmses, rs


# calcualte the value of the objective function
def evaluate_objective_function(masked_R, P, Q, s, beta, gamma):
    error = np.ma.dot(P, Q.T) - masked_R
    frobenius = np.linalg.norm(error)
    squared_error = frobenius * frobenius
    rmse = math.sqrt(squared_error / masked_R.count())
    cs = cut_size(graph_laplacian(adjacency(Q)), s)
    objective = squared_error + beta * (np.linalg.norm(P) ** 2 + np.linalg.norm(Q) ** 2) - \
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


# plot a heatmap
def heatmap(values, title):
    plt.figure(next(fc))
    sns.heatmap(values, xticklabels=False, yticklabels=False)
    plt.title(title)


if __name__ == '__main__':

    # figure counter
    fc = fcounter()

    #R = np.array([[5, 3, 0, 1],
    #              [4, 0, 0, 1],
    #              [1, 1, 0, 5],
    #              [1, 0, 0, 4],
    #              [0, 1, 5, 4]])

    PG, QG, R = generate_rating_matrix()

    n, m = R.shape
    k = 2
    steps = 500
    alpha = 0.00005  # learning rate
    beta = 0.1  # L2 regularization
    gamma = 0.1  # cut size regularization

    # division vector
    # s = np.array([1, -1, -1, 1])

    # random division vector
    # s = np.random.rand(m)
    # s[s > 0.5] = 1
    # s[s <= 0.5] = -1

    # exact division vector
    s = [-1] * int(m / 2)
    s.extend([1] * int(m / 2))

    # initialize P, Q with some random values
    P = np.random.rand(n, k)
    Q = np.random.rand(m, k)

    nP, nQ, objective, rmse, rs = matrix_factorization(R, P, Q, s, steps, alpha, beta, gamma)
    nR = np.dot(nP, nQ.T)

    print("approximated rating matrix")
    print(nR)

    print("rating matrix")
    print(R)

    print('adjacency')
    print(adjacency(Q))

    print('laplacian')
    print(graph_laplacian(adjacency(Q)))

    # plot the matrices
    heatmap(R, 'R')
    heatmap(nR, 'Fitted R')
    heatmap(PG, 'P')
    heatmap(P, 'Fitted P')
    heatmap(P - PG, 'Differences in P')
    print('max abs difference in P')
    print(np.max(np.absolute((P - PG))))
    heatmap(QG, 'Q')
    heatmap(Q, 'Fitted Q')
    heatmap(Q - QG, 'Differences in Q')
    print('max abs difference in Q')
    print(np.max(np.absolute((Q - QG))))

    print("final objective")
    print(objective[-1])
    print("final rmse")
    print(rmse[-1])
    print("final cut size")
    print(rs[-1])

    # plot the objective function and the rmse
    plot(objective, 'J', 'Objective Function')
    plot(rmse, 'RMSE', 'RMSE')
    plot(rs, 'R', 'Cut Size')
    plt.show()

