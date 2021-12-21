import sys

import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
from matplotlib.transforms import ScaledTranslation


def  create_third_block(trainX, trainy):
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
    return sol[:d]


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


def calc_errors(l, trainx, trainy, x, y, n, m):
    ave_error = 0
    min_err = sys.maxsize
    max_err = 0
    for i in range(n):
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
    ave_error /= n
    return ave_error, max_err, min_err


def plt_error(err_test, min_test, max_test, err_train, min_train, max_train, err_trainb, err_testb, nums):
    err_train = np.asarray(err_train)
    min_train = np.asarray(min_train)
    max_train = np.asarray(max_train)
    err_test = np.asarray(err_test)
    min_test = np.asarray(min_test)
    max_test = np.asarray(max_test)
    train, test = plt.subplots()
    trans1 = test.transData + ScaledTranslation(-5 / 72, 0, train.dpi_scale_trans)
    trans2 = test.transData + ScaledTranslation(+5 / 72, 0, train.dpi_scale_trans)
    test.errorbar(nums, err_train, [err_train - min_train, max_train - err_train], fmt='ok', lw=1,
                 linestyle="none", transform=trans1,ecolor='cyan')
    test.errorbar(nums, err_test, [err_test - min_test, max_test - err_test], fmt='ok', lw=1,
                 linestyle="none", transform=trans2, ecolor='blue')

    plt.plot(nums, err_train)
    plt.plot(nums, err_test)
    err_trainb = np.asarray(err_trainb)
    err_testb = np.asarray(err_testb)
    numsb = [1, 3, 5, 8]
    plt.plot(numsb, err_trainb, "s")
    plt.plot(numsb, err_testb, "s")
    plt.title("Average error")
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
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, testX, testy, 10, 100)
        errors_test.append(err)
        min_errors_test.append(min_err)
        max_errors_test.append(max_err)
        # calculate train error
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, trainX, trainy, 10, 100)
        errors_train.append(err)
        min_errors_train.append(min_err)
        max_errors_train.append(max_err)

    return errors_test, min_errors_test, max_errors_test, errors_train, min_errors_train, max_errors_train


def q2b():
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # run the softsvm algorithm for 10**n
    nums = [1, 3, 5, 8]
    lamdas = np.asarray([10 ** n for n in nums], dtype='int64')
    errors_train = []
    errors_test = []
    for l in lamdas:
        # calculate test error
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, testX, testy, 1, 1000)
        errors_test.append(err)

        # calculate train error
        (err, max_err, min_err) = calc_errors(l, trainX, trainy, trainX, trainy, 1, 1000)
        errors_train.append(err)

    errors_testa, min_errors_test, max_errors_test, errors_traina, min_errors_train, max_errors_train = q2a()
    plt_error(errors_testa,
              min_errors_test,
              max_errors_test,
              errors_traina,
              min_errors_train,
              max_errors_train,
              errors_train,
              errors_test, np.arange(1,11))
    return errors_train, errors_test


if __name__ == '__main__':
    simple_test()
    q2a()
    q2b()
