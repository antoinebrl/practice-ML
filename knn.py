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

    def eval(self, data):
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

    c = knn.eval(data)

    plt.plot(data[np.where(c==0),0], data[np.where(c==0),1], 'bo')
    plt.plot(data[np.where(c==1),0], data[np.where(c==1),1], 'ro')
    plt.plot(data[np.where(c==2),0], data[np.where(c==2),1], 'ko')
    plt.plot(data[np.where(c==3),0], data[np.where(c==3),1], 'go')
    plt.show()

