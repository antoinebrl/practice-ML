# Author : Antoine Broyelle
# Licence : MIT
# inspired by : KTH - DD2432 : Artificial Neural Networks and Other Learning Systems
# https://www.kth.se/student/kurser/kurs/DD2432?l=en

import numpy as np
import sys

class PCN:
    '''Perceptron. Based on McCulloch and Pitts neurons'''

    def __init__(self, inputs, targets, bias=True, delta=False):
        '''
        Constructor
            :param inputs : set of data points as row vectors
            :param targets : set of targets as row vectors. For 1D, the format could be a single row
            :param bias : deals with bias and non zero thresholds
            :param delta : use the delta learning rule or the Perceptron learning rule
        '''
        # Prerequisites
        if np.ndim(inputs) > 2:
            raise Exception('[pcn][__init__] The input should be a matrix with maximum 2 indexes')
        if np.shape(inputs)[0] != np.shape(targets)[0]:
            raise Exception('[pcn][__init__] The input and target matrices do not have the same '
                            'number of samples')

        # Parameters
        dimensions = np.shape(inputs)
        self.nbSamples = dimensions[0]
        self.dimIn = 1 if np.ndim(inputs) == 1 else dimensions[1]
        self.dimOut = 1 if np.ndim(targets) <= 1 else np.shape(targets)[1]
        self.delta = delta

        # Data
        self.targets = targets if np.ndim(targets) > 1 else np.transpose([targets])
        if delta:
            self.targets = np.where(self.targets > 0, 1, -1) # bipolar encoding

        if bias:
            self.inputs = np.concatenate((inputs, np.ones((self.nbSamples, 1))), axis=1)
        else:
            self.inputs = inputs

        # Init network
        self.weights = np.random.rand(self.dimIn + 1 * bias, self.dimOut) - 0.5 # mean = 0


    def predict(self, data):
        '''Recall. Compute the activation'''
        # sum
        activation = np.dot(data, self.weights)
        # activation/thresholding
        return np.where(activation > 0, 1, 0)


    def fwd(self):
        '''Forward step of the training process'''
        # sum
        activation = np.dot(self.inputs, self.weights)
        # activation/thresholding
        return np.where(activation > 0, 1, 0) if not self.delta else activation


    def train(self, eta=0.1, nbIte=10, batch=True):
        '''
        Training using back-propagation
            :param eta: learning rate for the hidden layer
            :param beta: learning rate for the output layer
            :param nbIte: number of iterations
            :param batch: use batch (synchronised) or on-line (asynchronised) learning
        '''
        if batch:
            for n in range(nbIte):
                activation = self.fwd()
                self.weights -= eta * np.dot(np.transpose(self.inputs), activation - self.targets)
                if np.mod(n,10) == 0:
                    print >> sys.stderr, "epoch: ", n
        else: # sequential
            for n in range(nbIte):
                M = np.shape(self.inputs)[1]
                for data in range(self.nbSamples):
                    for j in range(self.dimOut):
                        activation = 0
                        for i in range(M):
                            activation += self.inputs[data][i] * self.weights[i][j]

                        # Thresholding
                        if not self.delta :
                            if activation > 0:
                                activation = 1
                            else:
                                activation = 0

                        for i in range(M):
                            self.weights[i][j] -= eta * (activation - self.targets[data][j]) * self.inputs[data][i]
                if np.mod(n,10) == 0:
                    print >> sys.stderr, "epoch: ", n

        return self.predict(self.inputs)



if __name__ == "__main__":

    '''Logic Tests'''
    eta = 0.1
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    ANDtargets = np.array([[0], [0], [0], [1]])
    ORtargets = np.array([0, 1, 1, 1]) # second format for 1 dimensional targets
    XORtargets = np.array([0, 1, 1, 0]) # non linearly separable

    # AND
    print "AND"
    pcn = PCN(inputs, ANDtargets)
    output = pcn.train(eta, 20)
    print "Perceptron learning rule"
    print output

    pcn = PCN(inputs, ANDtargets, delta=True)
    output = pcn.train(eta, 20)
    print "Delta learning rule"
    print output

    # OR
    print "OR"
    pcn = PCN(inputs, ORtargets)
    output = pcn.train(eta, 20)
    print "Perceptron learning rule"
    print output

    pcn = PCN(inputs, ORtargets, delta=True)
    output = pcn.train(eta, 20)
    print "Delta learning rule"
    print output

    # XOR
    print "XOR"
    pcn = PCN(inputs, XORtargets)
    output = pcn.train(eta, 20)
    print "Perceptron learning rule"
    print output

    pcn = PCN(inputs, XORtargets, delta=True)
    output = pcn.train(eta, 20)
    print "Delta learning rule"
    print output

    '''2D test'''
    import pylab as plt

    dataA = np.random.multivariate_normal([-3, +1], [[1,0],[0,1]], 100)
    dataB = np.random.multivariate_normal([+3, -1], [[1,0],[0,1]], 100)
    data = np.concatenate((dataA, dataB))
    targets = np.concatenate((np.ones((100,1)), np.zeros((100,1))))
    # shuffle
    p = np.random.permutation(np.shape(data)[0])
    data = data[p]
    targets = targets[p]

    pcn = PCN(data, targets, delta=True, bias=False)
    output = pcn.train(nbIte=120)

    x = np.arange(-6, 6, 0.01)
    y = np.arange(-4, 4, 0.01)
    xx0, yy0 = np.meshgrid(x, y)
    xx = np.reshape(xx0, (xx0.shape[0]*xx0.shape[1],1))
    yy = np.reshape(yy0, (yy0.shape[0]*yy0.shape[1],1))
    grid = np.concatenate((xx,yy), axis=1)
    area = pcn.predict(grid)

    plt.scatter(data[:, 0], data[:, 1], c=output[:,0], s=120)
    plt.contour(xx0, yy0, area.reshape(xx0.shape))
    plt.show()

