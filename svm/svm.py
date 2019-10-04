import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data import inputs, targets, plotContour, plotPoints, plotShow
import time


def kernel_lin(x, y):
    return np.dot(x, y)


def kernel_poly(x, y):
    p = 3
    return (np.dot(x, y) + 1) ** p


def kernel_rbf(x, y):
    a = x - y
    sigma = 1
    return np.exp((np.sqrt(np.dot(a, a)) ** 2) / (2 * sigma ** 2))


class SVM:
    def __init__(self, C, kernel_function):
        self.kernel = kernel_function
        self.C = C

    def train(self, x, t):

        self.X = np.copy(x)
        self.T = np.copy(t)

        self.X.flags.writeable = False
        self.T.flags.writeable = False

        N = x.shape[0]

        P_ij = np.asanyarray(
            [
                [
                    self.T[i] * self.T[j] * self.kernel(self.X[i], self.X[j])
                    for j in range(N)
                ]
                for i in range(N)
            ]
        )

        P_ij.flags.writeable = False

        def objective(a):  # eq (4)
            return 1 / 2 * (a @ P_ij @ a.T) - np.sum(a)

        def zerofun(a):  # eq (10)
            return np.dot(a, self.T)

        start = np.zeros(N)
        B = [(0, None) for b in range(N)]
        XC = {"type": "eq", "fun": zerofun}

        mini = minimize(objective, start, bounds=B, constraints=XC)
        self.alpha = mini["x"]
        sv_args = np.argwhere(self.alpha > 10 ** -5).flatten()

        assert len(sv_args), "Found no alphas"

        # keep only non-zeros alphas, X, T (the support vectors)
        self.X = self.X[sv_args]
        self.T = self.T[sv_args]
        self.alpha = self.alpha[sv_args]

        # set B by forcing indicator of support vectors to be target
        self.b = 0
        self.b = self.indicator(self.X[0]) - self.T[0]

    def indicator(self, x):
        return (
            np.sum(
                self.alpha * self.T * np.array([self.kernel(x, xi) for xi in self.X])
            )
            - self.b
        )


svm = SVM(40, kernel_rbf)
svm.train(inputs, targets)
plotContour(svm.indicator)
plotPoints()
plotShow()

# print (np.array([[1,2,3],[1,2,3]]) @ np.array([1,2,3]).T)

# svs = inputs[np.argwhere(svm.alpha != 0)]

# plt.show(block=False)
# plt.pause(1)
# plt.plot(svs.T[0], svs.T[1], "o")
# plt.show()
