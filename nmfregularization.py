# Created by elex
# Implementation of regularized nonnegative matrix factorization. Script reuses some scikit-learn functions
# Goal: find two non-negative matrices (W, H), whose product approximates the non-negative matrix X.
# Jan 3 2018
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns

from matrixgenerator import generate_rating_matrix
from matrixgenerator import sample_ratings
from networks import katz_similarity


# graph regularized nmf
def regularized_nmf(R, A, Wr, Hr, lambd=0.0, max_iter=100):
        W = np.copy(Wr)
        H = np.copy(Hr)

        # mask the nans
        Rm = np.ma.array(R, mask=np.isnan(R))
        count = Rm.count()

        Rf = np.ma.filled(Rm, fill_value=0.0)

        # diagonal matrix of degrees
        D = np.diag(np.sum(A, axis=0))

        # book keeping
        rmses = []
        objectives = []

        eps = np.finfo(float).eps

        for i in range(max_iter):
            RtW = lambd * np.dot(A, H.T) + np.dot(Rf.T, W)
            HtWtW = lambd * np.dot(D, H.T) + np.dot(H.T, np.dot(W.T, W))
            H = np.multiply(H, np.divide(RtW + eps, HtWtW + eps).T)

            RHt = np.dot(Rf, H.T)
            WHHt = np.dot(W, np.dot(H, H.T))
            W = np.multiply(W, np.divide(RHt + eps, WHHt + eps))

            error = np.ma.dot(W, H) - Rm
            squared_error = np.sum(np.ma.diag(np.ma.dot(error.T, error)))
            rmse = math.sqrt(squared_error / count)
            rmses.append(rmse)
            objectives.append(squared_error + lambd * np.sum(np.diag(np.dot(H, np.dot(D - A, H.T)))))

        return W, H, rmses, objectives


# figure counter
def fcounter():
    i = 1
    while True:
        yield i
        i += 1


# plot values vs. iteration steps
def plot(values, ylabel, title):
    fig = plt.figure(next(fc))
    plt.plot(range(len(values)), values)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    fname = title.replace('$\lambda=$', 'lambda=')
    plt.savefig('./images/' + fname + '.png')
    plt.close(fig)


# plot a heatmap
def heatmap(values, title):
    fig = plt.figure(next(fc))
    sns.heatmap(values, xticklabels=False, yticklabels=False)
    plt.title(title)
    fname = title.replace('$\lambda=$', 'lambda=')
    plt.savefig('./images/' + fname + '.png')
    plt.close(fig)


def results(R, W, H, rmse, objective, plotting=False):
    print("W")
    print(W)
    print("H")
    print(H)
    print("reconstruction")
    Rr = np.dot(W, H)
    print(Rr)
    print('R')
    print(R)

    print('embedded distances')
    print(pairwise_distances(H.T))

    print('rmse')
    print(rmse)
    print('final rmse')
    print(rmse[-1])

    print('objective')
    print(objective)
    print('final objective')
    print(objective[-1])


def link_statistics(A):
    m, m = A.shape
    print('total number of links: ', np.sum(A))
    g11 = np.sum(A[0:int(m / 2), 0:int(m / 2)])
    g12 = np.sum(A[0:int(m / 2), int(m / 2):m])
    g21 = np.sum(A[int(m / 2):m, 0:int(m / 2)])
    g22 = np.sum(A[int(m / 2):m, int(m / 2):m])
    print('links within groups: ', g11 + g22)
    print('links between groups: ', g12 + g21)
    print('fraction links between groups: ', (g12 + g21) / (g12 + g21 + g11 + g22))


def ex1(plotting=False, verbose=True):
    n = 50
    m = 100
    k = 5
    fraction = 0.1

    # generate random matrices
    WG, HtG, R = generate_rating_matrix(n=n, m=m)
    R = sample_ratings(R, fraction=fraction)

    if plotting:
        heatmap(R, 'R')
        heatmap(WG, 'WG')
        heatmap(HtG.T, 'HG')

    # print the generated matrices
    if verbose:
        print('WG')
        print(WG)
        print('HtG')
        print(HtG)

    # normalize the ratings to unity
    Rm = np.ma.array(R, mask=np.isnan(R))
    length = np.sqrt(np.ma.diag(np.ma.dot(Rm.T, Rm)))
    Rm = Rm / length

    # similarity graph from the rating matrix
    G1 = np.ma.filled(np.ma.dot(Rm.T, Rm), fill_value=0.0)
    print(np.count_nonzero(G1))

    # similarity graph with a base similarity
    G2 = 0.01 + np.copy(G1)
    print(np.count_nonzero(G2))

    # rbf kernel applied to the similarity graph
    sigma = np.std(G1)
    print(sigma)
    rbf = lambda x: math.exp(-x / sigma)
    vrbf = np.vectorize(rbf)
    G3 = vrbf(1.0 - np.copy(G1))
    print(G3)

    # regular similarity graph from the rating matrix
    G4 = katz_similarity(np.copy(G1))

    # random initialization works better
    # we initialize once to compare the results
    Wr = np.random.rand(n, k)
    Hr = np.random.rand(m, k)
    Hr = Hr.T

    # fit -> higher lambda is stronger regularization
    i = 1
    for G in [G1, G2, G3, G4]:
        for lambd in np.arange(0.0, 21.0, 1.0):
            W, H, rmse, objective = regularized_nmf(R, G, Wr, Hr, lambd=lambd)
            print('graph: ', i)
            print('regularization:', lambd)
            print('final rmse: ', rmse[-1])
            print('final objective: ', objective[-1])

            # print results
            if verbose:
                results(R, W, H, rmse, objective, plotting=plotting)
                print('max abs difference in W')
                print(np.max(np.absolute((W - WG))))
                print('max abs difference in H')
                print(np.max(np.absolute((H.T - HtG))))

            # construct A to compare adjacency matrices with each other
            A = np.dot(H.T, H)
            np.fill_diagonal(A, 0.0)
            link_statistics(A)

            # print matrix and cut size
            if verbose:
                print('A')
                print(A)

            if plotting:
                # plot matrices
                heatmap(np.dot(W, H), 'Reconstructed_R_%d_$\lambda=$%2.1f' % (i, lambd))
                heatmap(W, 'W_%d_$\lambda=$%f' % (i, lambd))
                heatmap(H.T, 'H_%d_$\lambda=$%f' % (i, lambd))

                # plot rmse & objective
                plot(rmse, 'RMSE', 'RMSE_%d_$\lambda=$%f' % (i, lambd))
                plot(objective, 'Objective Function', 'Objective_Function_%d_$\lambda=$%2.1f' % (i, lambd))
                heatmap(A, 'adjacency_%d_$\lambda=$%f' % (i, lambd))

            print('---------------------------')

        i = i + 1
        print('**********************************')


if __name__ == '__main__':

    plotting = True
    verbose = False

    # figure counter
    fc = fcounter()

    # run
    ex1(plotting=plotting, verbose=verbose)
