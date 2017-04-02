# Author : Antoine Broyelle
# Licence : MIT
# inspired by : KTH - DD2432 : Artificial Neural Networks and Other Learning Systems
# https://www.kth.se/student/kurser/kurs/DD2432?l=en

import numpy as np
from kmeans import Kmeans
from pcn import PCN
from utils.distances import euclidianDist

class RBF:
    '''Radial Basis Function Network. Can be used for classification or function approximation'''

    def __init__(self, inputs, targets, n=1, sigma=0, distance=euclidianDist,
                 weights=None, usage='class', normalization=False):
        '''
        :param inputs: set of data points as row vectors
        :param targets: set of targets as row vectors
        :param n: (int) number of weights.
        :param sigma: (float) spread of receptive fields
        :param distance: (function) compute metric between points
        :param weights: set of weights. If None, weights are generated with K-means algorithm.
                Otherwise provided weights are used no matter the value of n.
        :param usage: (string) Should be equal to 'class' for classification and 'fctapprox' for
                function approximation. Otherwise raise an error.
        :param normalization: (bool) If true, perform a normalization of the hidden layer.
        '''
        if not usage is 'class' and not usage is 'fctapprox':
            raise Exception('[RBF][__init__] the usage is unrecognized. Should be equal to '
                            '"class" for classification and "fctapprox" for function approximation')

        self.targets = targets
        self.inputs = inputs
        self.dist = distance
        self.n = n
        self.weights = weights
        self.usage = usage
        self.normalization = normalization

        if sigma == 0:
            self.sigma = (inputs.max(axis=0)-inputs.min(axis=0)).max() / np.sqrt(2*n)
        else:
            self.sigma = sigma

    def fieldActivation(self, inputs, weights, sigma, dist):
        hidden = dist(inputs, weights)
        hidden = np.exp(- hidden / sigma)
        return hidden

    def train(self, nbIte=100):
        if self.weights is None:
            km = Kmeans(self.inputs, k=self.n, distance=self.dist)
            km.train(nbIte=1000)
            self.weights = km.centers

        hidden = self.fieldActivation(self.inputs, self.weights, self.sigma, self.dist)
        if self.normalization:
            hidden = hidden / np.sum(hidden, axis=1)[:, np.newaxis]

        if self.usage is 'class':
            self.pcn = PCN(inputs=hidden, targets=self.targets, delta=True)
            return self.pcn.train(nbIte=nbIte)
        else : # linear regression
            self.weights2 = np.linalg.inv(np.dot(hidden.T, hidden))
            self.weights2 = np.dot(self.weights2, np.dot(hidden.T, self.targets))
            return  np.dot(hidden, self.weights2)

    def predict(self, data):
        h = self.fieldActivation(data, self.weights, self.sigma, self.dist)
        if self.usage is 'class':
            return self.pcn.predict(h)
        else:
            return np.dot(h, self.weights2)


if __name__ == "__main__":

    # Classification
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    XORtargets = np.array([[0], [1], [1], [0]])
    rbf = RBF(inputs=inputs, targets=XORtargets, n=4)
    print rbf.train(nbIte=300)

    # Function approximation
    import matplotlib.pyplot as plt
    x = np.linspace(start=0, stop=2*np.pi, num=63)
    y = np.sin(x)
    w = np.linspace(start=0, stop=2 * np.pi, num=8)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    w = w[:, np.newaxis]

    rbf = RBF(inputs=x, targets=y, usage='fctapprox', weights=w, normalization=True)
    out = rbf.train()

    plt.plot(x,y, 'r')
    plt.plot(x,out, 'b')
    plt.show()

