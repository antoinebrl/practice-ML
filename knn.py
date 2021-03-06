# Author : Antoine Broyelle
# Licence : MIT
# inspired by : ENSIMAG - 4MM2SIRR : Systemes intelligents: reconnaissance et raisonnement
# http://ensimag.grenoble-inp.fr/cursus-ingenieur/syst-egrave-mes-intelligents-reconnaissance-et-raisonnement-en-anglais-4mm2sirr-539495.kjsp

import numpy as np

class KNN:
    '''K-Nearest Neighbors classifier'''

    def __init__(self, inputs, targets, k=1):
        if k < 1:
            raise Exception("[KNN][init] k must be greater than zero")
        self.inputs = inputs
        self.targets = targets
        self.k = k

    def train(self):
        return

    def predict(self, data):
        distances = np.sum((data[:, np.newaxis] - self.inputs)**2, axis=2)
        labelNearests = np.take(self.targets, np.argsort(distances)[:,0:self.k])
        freq = np.apply_along_axis(np.bincount, axis=1, arr=labelNearests,
                                   minlength=np.max(self.targets)+1) # Use lot of memory
        return freq.argmax(axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataA = np.random.multivariate_normal([2, -2], [[1,0],[0,1]], 500)
    dataB = np.random.multivariate_normal([-2, 2], [[1,0],[0,1]], 500)
    dataC = np.random.multivariate_normal([2, 2], [[1,0],[0,1]], 500)
    dataD = np.random.multivariate_normal([-2, -2], [[1,0],[0,1]], 500)
    data = np.concatenate((dataA, dataB, dataC, dataD))
    # shuffle
    p = np.random.permutation(np.shape(data)[0])
    data = data[p]

    training = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
    classes = np.array([[0],[1],[2],[3]])
    knn = KNN(training, classes, k=1)

    c = knn.predict(data)

    x = np.arange(-6, 6, 0.01)
    y = np.arange(-4, 4, 0.01)
    xx0, yy0 = np.meshgrid(x, y)
    xx = np.reshape(xx0, (xx0.shape[0]*xx0.shape[1],1))
    yy = np.reshape(yy0, (yy0.shape[0]*yy0.shape[1],1))
    grid = np.concatenate((xx,yy), axis=1)
    area = knn.predict(grid)

    plt.scatter(data[:,0], data[:,1], c=c, s=30)
    plt.contour(xx0, yy0, area.reshape(xx0.shape))
    plt.show()

