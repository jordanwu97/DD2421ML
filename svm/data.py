import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)


def generateData(classes_stats, num_points_per_class):

    datapoints = np.ndarray((0, 2))
    labels = np.ndarray((0))

    for label, means, variance in classes_stats:
        datapoints = np.concatenate(
            (datapoints, np.random.randn(num_points_per_class, 2) * variance + means)
        )
        labels = np.concatenate((labels, np.ones(num_points_per_class) * label))

    return datapoints, labels


def plotPoints(data, label):

    A_args = np.argwhere(label == 1).flatten()
    B_args = np.argwhere(label == -1).flatten()

    A = data[A_args]
    B = data[B_args]

    minX, maxX = np.min(data.T[0]) - 1, np.max(data.T[0]) + 1
    minY, maxY = np.min(data.T[1]) - 1, np.max(data.T[1]) + 1

    plt.plot(A.T[0], A.T[1], "r.")
    plt.plot(B.T[0], B.T[1], "b.")

    plt.xlim((minX, maxX))
    plt.ylim((minY, maxY))
    return plt


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
    return plt


if __name__ == "__main__":
    stats = [(1, (1.5, 0.5), 0.1), (1, (-1.5, 0.5), 0.1), (-1, (0.0, -0.5), 0.1)]
    for _ in range(10):
        d, l = generateData(stats, 10)
        plotPoints(d, l).show(block=False)
        plt.pause(0.5)
        plt.clf()

