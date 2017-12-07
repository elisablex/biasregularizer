# Created by elex
# Implementation of matrix factorization
# Nov 21 2017

import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import linalg as LA


# Input:
#   r ... rating matrix
#   p ... represents strength of association between user and features, |U| x K
#   q ... represents strength of association between items and feautures |I| x K
#   k ... latent features
#   steps ... max number of steps to perform optimization
#   alpha ... constant, learning rate (rate of approaching the minimum with gradient descent)
#   beta ... regularization parameter, controls magnitude of user and item features so that P and Q
#           are good approximations of R without containing large numbers
# Output:
#   P and Q
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta = 0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e= e + pow(R[i][j] - np.dot(P[i,:], Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * pow(P[i][k],2) + pow(Q[k][j],2)
        if e < 0.001:
            break
    return P, Q.T


# Input:
#   matrix ... similarity matrix
#   normalize ... boolean value to define wether degrees should be normalized
# Output:
#   combinatorialLaplacian ... graph laplacian
def laplacian(matrix, normalize=False):
    numRows, numCols = matrix.shape
    degrees = np.sum(matrix, axis=0)

    if normalize:
        normalizedDegrees = 1 / np.sqrt(np.array(
            [[degrees[i] * degrees[j] * matrix[i, j] for j in range(numCols)] for i in range(numRows)]
        ))

        # remove inf's created by dividing by zero
        normalizedDegrees[normalizedDegrees == np.inf] = 0

        return np.diag(np.ones(numRows)) - normalizedDegrees
    else:
        combinatorialLaplacian = np.diag(degrees) - matrix
        return combinatorialLaplacian

def bestCut(graph):
    laplacianMatrix = laplacian(graph, normalize=True)
    n, m = laplacianMatrix.shape

    eigenvalues, eigenvectors = np.linalg.eig(laplacianMatrix)

    # sort eigenvectors by eigenvalue increasing
    sortedIndices = eigenvalues.argsort()
    eigenvalues = eigenvalues[sortedIndices]
    eigenvectors = eigenvectors[:, sortedIndices]
    print(eigenvectors)

    # sort vertices of G by their value in the second eigenvector
    secondEigenvector = eigenvectors[:, 1]
    sortedVertexIndices = secondEigenvector.argsort()

    def cutQuality(j):
        firstHalf, secondHalf = sortedVertexIndices[range(j+1)], sortedVertexIndices[range(j+1, n)]
        firstTotal, secondTotal, crossTotal = 0, 0, 0

        for u in range(n):
            for v in range(m):
                if graph[u, v] > 0:
                    if u in firstHalf and v in firstHalf:
                        firstTotal += graph[u, v]
                    elif u in secondHalf and v in secondHalf:
                        secondTotal += graph[u, v]
                    else:
                        crossTotal += graph[u, v]

        if 0 == min(firstTotal, secondTotal):
            return np.inf

        return crossTotal / min(firstTotal, secondTotal)

    bestCutIndex = min(range(n), key=cutQuality)
    leftHalf, rightHalf = sortedVertexIndices[:bestCutIndex], sortedVertexIndices[bestCutIndex:]
    return list(sorted(leftHalf)), list(sorted(rightHalf))


# function that returns df/dQ
def model(P,Q,R):
    print("Matrix norm")
    print(2 * LA.norm(P.dot(Q.T) - R)) #scalar
    #dfdQ = 2 * LA.norm(P.matmul(Q.T) - R) * Q - 2 * X * S.matmul(S.T) * Q
    #return dfdQ

def cutsizeregularization(graph):
    print()

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

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)


    nP, nQ = matrix_factorization(R, P, Q, K)

    weightedMatrix = np.dot(nQ,nQ.T)
    laplacianMatrix = laplacian(weightedMatrix)
    print(laplacianMatrix)

    theBestCut = bestCut(weightedMatrix)
    print(theBestCut)

    #model(P, Q, R)
