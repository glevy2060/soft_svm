import sys

import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def create_third_block(trainX, trainy):
    block_matrix = np.asarray([trainy[0] * trainX[0]])
    for i in range(1, len(trainy)):
        vector = trainy[i] * trainX[i]
        block_matrix = np.append(block_matrix, [vector], axis=0)
    return block_matrix


def softsvm(l, trainX: np.array, trainy: np.array):
    d = len(trainX[0])
    m = len(trainX)

    # create H
    diag = np.zeros((d, d), dtype='int64')
    np.fill_diagonal(diag, 2 * l)
    H = np.block([[diag, np.zeros((d, m))], [np.zeros((m, d)), np.zeros((m, m))]])
    H = matrix(H)

    # create u
    u = np.append(np.zeros((d, 1)), (1 / m) * np.ones((m, 1)))
    u = matrix(u)

    # create v
    v = np.append(np.zeros((m, 1)), np.ones((m, 1)))
    v = matrix(v)

    # create A
    third_block = create_third_block(trainX, trainy)
    A = np.block([[np.zeros((m, d)), np.eye(m, m)], [third_block, np.eye(m, m)]])
    A = matrix(A)

    sol = np.asarray(solvers.qp(H, u, -A, -v)["x"])
    return sol[:784]


def simple_test():
    # load question 2 data
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def calc_errors(l, trainx, trainy, x, y):
    ave_error = 0
    min_err = sys.maxsize
    max_err = 0
    m = 100
    for i in range(10):
        # Get a random m training examples from the training set
        indices = np.random.permutation(trainx.shape[0])
        _trainX = trainx[indices[:m]]
        _trainY = trainy[indices[:m]]

        w = softsvm(l, _trainX, _trainY)
        pred = np.sign(x @ w)
        err = np.mean(pred.flatten() != y)
        min_err = min(min_err, err)
        max_err = max(max_err, err)
        ave_error += err
    ave_error /= 10
    return ave_error, max_err, min_err


def plt_error(err_test, min_test, max_test, err_train, min_train, max_train, nums):
    err_train = np.asarray(err_train)
    min_train = np.asarray(min_train)
    max_train = np.asarray(max_train)
    err_test = np.asarray(err_test)
    min_test = np.asarray(min_test)
    max_test = np.asarray(max_test)
    plt.errorbar(nums, err_train, [err_train - min_train, max_train - err_train], fmt='ok', lw=1,
                 ecolor='cyan')
    plt.errorbar(nums, err_test, [err_test - min_test, max_test - err_test], fmt='ok', lw=1,
                 ecolor='blue')
    plt.title("average test error")
    plt.xlabel("Power of 10")
    plt.ylabel("Average error")
    plt.legend(["Train Error", "Test Error"])
    plt.show()


def q2a():
    # load question 2 data
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # run the softsvm algorithm for 10**n
    nums = np.arange(1, 11)
    lamdas = np.asarray([10 ** n for n in nums], dtype='int64')
    errors_train = []
    min_errors_train = []
    max_errors_train = []
    errors_test = []
    min_errors_test = []
    max_errors_test = []
    for l in lamdas:
        # calculate test error
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, testX, testy)
        errors_test.append(err)
        min_errors_test.append(min_err)
        max_errors_test.append(max_err)
        # calculate train error
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, trainX, trainy)
        errors_train.append(err)
        min_errors_train.append(min_err)
        max_errors_train.append(max_err)

    plt_error(errors_test, min_errors_test, max_errors_test, errors_train, min_errors_train, max_errors_train, nums)



if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # x = np.asarray([[1],[2],[3]])
    # y = np.asarray([0,1,1])
    # l = 4
    # test = np.asarray([1,2])
    # test2 = np.asarray([3,4])
    # test = np.append([test] ,[test2], axis =0)
    # print(softsvm(l, x, y))
    q2a()
    # simple_test()

    # here you may add any code that uses the above functions to solve question 2
