# Created by elex
# Generator for synthetic datasets
# Dec 21 2017

import numpy as np
import matplotlib.pyplot as plt



def create_matrix(rows, cols, meanhigh, stdhigh, meanlow, stdlow):

    matrix = []

    for i in range(rows):
        randomhigh = generate_preferences(meanhigh, stdhigh)
        randomlow = generate_preferences(meanlow, stdlow)
        temp = []
        if i < int(rows/2):
            temp.append(randomhigh)
            temp.append(randomlow)
        else:
            temp.append(randomlow)
            temp.append(randomhigh)
        matrix.append(temp)


    return np.array(matrix)


def generate_preferences(mean, std):
    high = np.random.normal(mean, std)
    return high


def compute_rating_matrix(P, Q):
    R = np.dot(P,Q.T)
    return R


if __name__ == '__main__':

    k=2
    n=50 # number of users
    m=100 # number of items
    highprefmean = 5.0
    lowprefmean = 0.5
    highprefstd = 1.0
    lowprefstd = 0.1

    P = create_matrix(n, k, highprefmean, highprefstd, lowprefmean, lowprefstd)
    Q = create_matrix(m, k, highprefmean, highprefstd, lowprefmean, lowprefstd)
    R = compute_rating_matrix(P,Q)

    print("user matrix")
    print(P)
    print("item matrix")
    print(Q)
    print("rating matrix")
    print(R)

    plt.imshow(R, cmap='hot', interpolation='nearest')
    plt.show()