import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def generateCircularData():
    stats = [(-1, (0.0, 0.0), 20)]
    inputs, _ = generateData(stats, 50)
    neg_args = np.argwhere( np.linalg.norm(inputs,axis=1) < 20 ).flatten()
    pos_args = np.argwhere( np.linalg.norm(inputs,axis=1) > 30 ).flatten()

    inputs = np.concatenate((inputs[pos_args], inputs[neg_args]))
    targets = np.concatenate((np.ones(len(pos_args)), -1 * np.ones(len(neg_args))))
    return inputs, targets


def plotPoints(data, label, plot=plt):

    A_args = np.argwhere(label == 1).flatten()
    B_args = np.argwhere(label == -1).flatten()

    A = data[A_args]
    B = data[B_args]

    minX, maxX = np.min(data.T[0]) - 1, np.max(data.T[0]) + 1
    minY, maxY = np.min(data.T[1]) - 1, np.max(data.T[1]) + 1

    plot.plot(A.T[0], A.T[1], "r.")
    plot.plot(B.T[0], B.T[1], "b.")

    plot.xlim((minX, maxX))
    plot.ylim((minY, maxY))
    return plot


def plotContour(indicator, plot=plt):
    x_low, x_high = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    
    xgrid = np.linspace(x_low, x_high)
    ygrid = np.linspace(y_low, y_high)

    grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])

    plot.contourf(
        xgrid,
        ygrid,
        grid,
        (-1.0, 0.0, 1.0),
        # colors=("red", "black", "blue"),
        # linewidths=(1, 3, 1),
    )
    return plot

def plotContours(indicators, labels):
    x_low, x_high = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    
    xgrid = np.linspace(x_low, x_high)
    ygrid = np.linspace(y_low, y_high)

    colors = cm.get_cmap('rainbow')(np.linspace(0,1,len(indicators)))

    for label, ind, color in zip(labels, indicators, colors):
        print (color.reshape(-1,4))
        grid = np.array([[ind(np.array([x, y])) for x in xgrid] for y in ygrid])
        cs = plt.contour(
            xgrid,
            ygrid,
            grid,
            (0),
            colors=(color.reshape(-1,4)),
        )
        plt.clabel(cs, fmt={0:str(label)})
    return plt
