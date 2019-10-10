import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from data import *
import time


def kernel_lin(x, y):
    return np.dot(x, y)


def get_kernel_poly(p):
    def kernel_poly(x, y):
        return (np.dot(x, y) + 1) ** p

    return kernel_poly


def get_kernel_rbf(sigma):
    def kernel_rbf(x, y):
        return np.exp(-1 * (np.linalg.norm(x - y) ** 2) / (2 * sigma ** 2))

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
        print (self.alpha)
        assert np.max(self.alpha) > 0, "No alphas > 0"

        # set B by forcing indicator of support vectors to be target
        C = np.inf if self.C == None else self.C
        sv1_arg = np.argwhere((0 < self.alpha) & (self.alpha < C)).flatten()
        assert (len(sv1_arg)) > 0, "No support vectors found"
        sv1_arg = sv1_arg[0]

        self.b = 0
        self.b = self.indicator(self.X[sv1_arg]) - self.T[sv1_arg]

    def indicator(self, x):
        return (
            np.sum(
                self.alpha * self.T * np.array([self.kernel(x, xi) for xi in self.X])
            )
            - self.b
        )

    def getAccuracy(self):
        pred = np.where(np.array([self.indicator(x) for x in self.X]) >= 0, 1, -1)
        return np.sum(pred == self.T) / len(self.T)


if __name__ == "__main__":

    # Linear
    stats = [(1, (1.5, 0.5), 0.2), (1, (-1.5, 0.5), 0.2), (-1, (0.0, -0.5), 0.2)]
    inputs, targets = generateData(stats, 10)

    # svm = SVM(None, kernel_lin)
    # svm.train(inputs, targets)
    # print ("Accuracy:", svm.getAccuracy())
    # plotPoints(inputs, targets)
    # plt.title("Linear Kernel")
    # plotContour(svm.indicator)

    # stats = [(1, (0, 0), 1), (-1, (0.25, 0), 1)]
    # inputs, targets = generateData(stats, 50)
    # svm = SVM(None, kernel_lin)
    # svm.train(inputs, targets)
    # print ("Accuracy:", svm.getAccuracy())
    # plotPoints(inputs, targets)
    # plt.title("Linear Kernel")
    # plotContour(svm.indicator)
    # plt.savefig('pictures/unseperable.png', bbox_inches='tight')

    # plt.clf()

    # inputs, targets = generateCircularData()
    # svm = SVM(None, kernel_lin)
    # svm.train(inputs, targets)
    # print ("Accuracy:", svm.getAccuracy())
    # plotPoints(inputs, targets)
    # plt.title("Linear Kernel")
    # plotContour(svm.indicator)
    # plt.savefig('pictures/unseperable_circle.png', bbox_inches='tight')

    # exit()

    # for i, c in enumerate([0.1,1,10,100]):
    #     svm = SVM(c, kernel_lin)
    #     svm.train(inputs, targets)
    #     plt.subplot(2,2,i+1)
    #     plotPoints(inputs, targets)
    #     plotContour(svm.indicator)
    #     plt.title(f"Linear Kernel (C={c})")
    #     plt.tight_layout()
    
    # plt.savefig("pictures/various_c_values.png", bbox_inches='tight')

    # exit()

    # Poly
    # stats = [(1, (1.5, 0.5), 0.2), (1, (-1.5, 0.5), 0.2), (-1, (0.0, -0.5), 0.2)]
    # inputs, targets = generateData(stats, 10)
    # inds = []
    # for p in range(1,5):
    #     svm = SVM(None, get_kernel_poly(p))
    #     svm.train(inputs, targets)
    #     accuracy = svm.getAccuracy()
    #     inds.append(svm.indicator)

    # plotPoints(inputs, targets)
    # plotContours(inds, range(1,5))
    # plt.title("Decision Boundary vs. Various Polynomial Degree")
    # plt.savefig(f"pictures/poly.png", bbox_inches='tight')

    # # RBF
    inputs, targets = generateCircularData()
    sigmas = [5,8,10,15]
    plt.clf()
    for i, sig in enumerate(sigmas):
        try:
            ax = plt.subplot(2, 2, i + 1)
            plotPoints(inputs, targets)
            svm = SVM(np.inf, get_kernel_rbf(sig))
            svm.train(inputs, targets)
            plotContour(svm.indicator)
            ax.set_title(f"sigma={sig} accuracy={svm.getAccuracy()}")
            plt.axis("off")
            plt.tight_layout()
        except AssertionError:
            pass

    plt.savefig(f"pictures/rbf_circle.png", bbox_inches='tight')
    

    stats = [(1, (1.5, 0.5), 0.2), (1, (-1.5, 0.5), 0.2), (-1, (0.0, -0.5), 0.2)]
    inputs, targets = generateData(stats, 10)
    sigmas = [0.2,0.4,0.6,0.8]
    plt.clf()
    for i, sig in enumerate(sigmas):
        try:
            ax = plt.subplot(2, 2, i + 1)
            plotPoints(inputs, targets)
            svm = SVM(np.inf, get_kernel_rbf(sig))
            svm.train(inputs, targets)
            plotContour(svm.indicator)
            ax.set_title(f"sigma={sig} accuracy={svm.getAccuracy()}")
            plt.axis("off")
            plt.tight_layout()
        except AssertionError:
            pass

    plt.savefig(f"pictures/rbf_original.png", bbox_inches='tight')