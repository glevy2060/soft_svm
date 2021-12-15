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

    alpha = np.asarray(solvers.qp(H, u, -A, -v)["x"])
    alpha = alpha[:m]
    return alpha

def softsvmbfWithG(l: float, sigma: float, trainX: np.array, trainy: np.array, G):
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
    third_block = third_block.reshape(len(third_block), len(third_block))
    A = np.block([[np.zeros((m, m)), np.eye(m, m)], [third_block, np.eye(m, m)]])
    A = matrix(A)

    alpha = np.asarray(solvers.qp(H, u, -A, -v)["x"])
    alpha = alpha[:m]
    return alpha

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


def q4a():
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    plt.scatter(trainX.T[:1], trainX.T[1:], s=3, c=trainy)
    plt.title("Training Points By Color")
    plt.show()


def cross_validation(trainx, trainy, params, k):
    s = np.asarray(list(zip(trainx, trainy)))
    si = np.split(s, k)
    for j in range (len(params)):  # params[lambda, sigma]
        for i in range(k):
            v = si[i]  # chosen set
            sTag = np.delete(si, i, 0)  # create s'= s\si = 4 groups in size 400*2
            sTagx = sTag.reshape(1600, 2)[:, 0]  # take only X
            sTagy = sTag.reshape(1600, 2)[:, 1]  # take only Y
            G = rbf(params[j][1], sTagx)
            hi = softsvmbfWithG(params[j][0], params[j][1], sTagx, sTagy, G)
            vx = v[:, 0]
            vy = v[:, 1]


def q4b():
    lamda = np.asarray([1, 10, 100])
    sigma = np.asarray([0.01, 0.5, 1])
    params = np.asarray(list(zip(lamda, sigma)))
    data = np.load('EX2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    sol = []
    cross_validation(trainX, trainy, params, 5)


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # q4a()
    q4b()

    # here you may add any code that uses the above functions to solve question 4
