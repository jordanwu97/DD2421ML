import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

classA = np.concatenate(
    (
        np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
        np.random.randn(10, 2) * 0.2 + [-1.5, 0.5],
    )
)

classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))

targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0]  # Number of rows (samples)

permute = list(range(N))
np.random.shuffle(permute)

inputs = inputs[permute, :]
targets = targets[permute]


def plotPoints():
    plt.plot(classA.T[0], classA.T[1], "r.")
    plt.plot(classB.T[0], classB.T[1], "b.")

def plotContour(indicator):
    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-5, 5)

    grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])

    plt.contour(
        xgrid,
        ygrid,
        grid,
        (-1.0, 0.0, 1.0),
        colors=("red", "black", "blue"),
        linewidths=(1, 3, 1),
    )

def plotShow():
    plt.show()
