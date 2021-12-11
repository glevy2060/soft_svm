import math

import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt

def calc_k(xi, xj, sigma):
    xij = xi-xj
    norm = np.linalg.norm(xij)
    power = (norm**2) / (2*sigma)
    k = math.exp(-power)
    return k


def rbf(sigma: float, trainX: np.array):
    G = np.asarray([[]])
    for i in range(len(trainX)):
        curr = []
        for j in range(len(trainX)):
            curr.append(calc_k(trainX[i], trainX[j], sigma))
        curr = np.asarray(curr)
        if i == 0:
            G = [curr]
        else:
            G = np.append(G, [curr], axis=0)
    return G


def create_third_block(G, trainy):
    block_matrix = np.asarray([[]])
    for i in range(0, len(G)):
        curr = []
        for j in range(len(G)):
            curr.append(trainy[i] * G[i][j])
        curr = np.asarray(curr)
        if i == 0:
            block_matrix = [curr]
        else:
            block_matrix = np.append(block_matrix, [curr], axis=0)
    return block_matrix


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    G = rbf(sigma, trainX) #*m*m
    d = len(trainX[0]) #784
    m = len(trainX)

    # create H
    H = np.block([[2*l*G, np.zeros((m, m))], [np.zeros((m, m)), np.zeros((m, m))]])
    H = matrix(H)

    # create u
    u = np.append(np.zeros((m, 1)), (1 / m) * np.ones((m, 1)))
    u = matrix(u)

    # create v
    v = np.append(np.zeros((m, 1)), np.ones((m, 1)))
    v = matrix(v)

    # create A
    third_block = create_third_block(G, trainy)
    A = np.block([[np.zeros((m, m)), np.eye(m, m)], [third_block, np.eye(m, m)]])
    A = matrix(A)

    sol = np.asarray(solvers.qp(H, u, -A, -v)["x"])
    return sol[:784]

    raise NotImplementedError()


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"



if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    # x = np.asarray([[1],[2],[3]])
    # sigma = 2
    # rbf(sigma, x)

    # here you may add any code that uses the above functions to solve question 4
