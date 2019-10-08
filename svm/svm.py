import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data import generateData, plotContour, plotPoints
import time


def kernel_lin(x, y):
    return np.dot(x, y)


def get_kernel_poly(p):
    def kernel_poly(x, y):
        return (np.dot(x, y) + 1) ** p

    return kernel_poly


def get_kernel_rbf(sigma):
    def kernel_rbf(x, y):
        a = x - y
        sigma = 1
        return np.exp((np.sqrt(np.dot(a, a)) ** 2) / (2 * sigma ** 2))

    return kernel_rbf


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
        B = [(0, self.C) for b in range(N)]
        XC = {"type": "eq", "fun": zerofun}

        mini = minimize(objective, start, bounds=B, constraints=XC)
        self.alpha = mini["x"]
        self.alpha = np.where(self.alpha > 10 ** -5, self.alpha, 0)

        assert np.sum(self.alpha) > 0, "No alphas > 0"

        # keep only non-zeros alphas, X, T (the support vectors)

        # set B by forcing indicator of support vectors to be target
        sv1_arg = np.argwhere((0 < self.alpha) & (self.alpha < self.C)).flatten()
        print (self.alpha)
        assert (len(sv1_arg)) > 0, "No support vectors found"
        sv1_arg = sv1_arg[0]

        self.b = 0
        self.b = self.indicator(self.X[sv1_arg]) - self.T[sv1_arg]

        # Keep only samples where alpha > 0
        keep = np.argwhere(self.alpha > 0).flatten()
        self.X = self.X[keep]
        self.T = self.T[keep]
        self.alpha = self.alpha[keep]

    def indicator(self, x):
        return (
            np.sum(
                self.alpha * self.T * np.array([self.kernel(x, xi) for xi in self.X])
            )
            - self.b
        )


if __name__ == "__main__":
    stats1 = [(1, (1.5, 0.5), 0.2), (1, (-1.5, 0.5), 0.2), (-1, (0.0, -0.5), 0.2)]
    
    stats2 = [(1, (0,5), 1), (-1, (0,0), 1)]
    
    inputs, targets = generateData(stats2, 10)

    svm = SVM(0.05, kernel_lin)
    svm.train(inputs, targets)
    plotContour(svm.indicator)
    plotPoints(inputs, targets).show()