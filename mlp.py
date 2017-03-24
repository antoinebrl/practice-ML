# Author : Antoine Broyelle
# Licence : MIT
# inspired by : KTH - DD2432 : Artificial Neural Networks and Other Learning Systems
# https://www.kth.se/student/kurser/kurs/DD2432?l=en

import numpy as np
import pylab as pl

class MLP:
    '''Multi-layers Perceptron.'''

    def __init__(self, inputs, targets, nbNodes=1, outputType='logic'):
        '''
        Constructor
            :param inputs: set of data points as row vectors
            :param nbNodes: number of hidden nodes
            :param outputType: can be 'logic' with a sigmoid, 'linear', or 'softmax'
        '''
        # Prerequisites
        if np.ndim(inputs) > 2:
            raise Exception('[pcn][__init__] The input should be a matrix with maximun 2 indexes')
        if np.shape(inputs)[0] != np.shape(targets)[0]:
            raise Exception('[pcn][__init__] The input and target matrixs do not have the same number of samples')

        # Parameters
        dimensions = np.shape(inputs)
        self.nbSamples = dimensions[0]
        self.dimIn = 1 if np.ndim(inputs) == 1 else dimensions[1]
        self.dimOut = 1 if np.ndim(targets) <= 1 else np.shape(targets)[1]
        self.nbNodes = nbNodes
        self.outputType = outputType

        # Data
        self.targets = targets
        self.inputs = np.concatenate((inputs, np.ones((self.nbSamples, 1))), axis=1)

        # Initialise network
        # uniform distribution of weigths in [-1/sqrt(n), 1/sqrt(n)] with n number of input node
        self.w1 = 2*(np.random.rand(self.dimIn + 1, self.nbNodes) - 0.5) / np.sqrt(self.dimIn)
        self.w2 = 2*(np.random.rand(self.nbNodes + 1, self.dimOut) - 0.5) / np.sqrt(self.nbNodes)


    def __addColumn(self, inputs):
        return np.concatenate((inputs, np.ones((np.shape(inputs)[0],1))),axis=1)


    def __phi(self,x):
        '''Sigmoid function for activation'''
        return 1.0 / (1.0 + np.exp(-0.8 * x))

    def __deltaPhi(self,x):
        '''Derivative of the Sigmoid function phi'''
        return 0.8 * np.exp(-0.6 * x) * self.__phi(x)**2


    def predict(self, inputs=None, training=False):
        '''
        Recall/Forward step of the back-propagation algorithm
            :param inputs:
            :param training: if called with training = True, temporary calculations are returned
            :return: In case training = True :
                        oout: output of the network. oout = phi(oin)
                        oin: input of output nodes. oin = hout*W2
                        hout : output of the first layer. hout = phi(hin)
                        hin : intput of the hidden nodes. hin = inputs*W1
                    Otherwise : oout
            :warn: be careful with matrix dimensions due to the bias terms
        '''
        if inputs is None:
            inputs = self.inputs
        else:
            inputs = self.__addColumn(inputs)

        hin = np.dot(inputs, self.w1)
        hout = self.__phi(hin)

        oin = np.dot(self.__addColumn(hout), self.w2)

        if self.outputType == 'linear':
            result =  oin, oin, hout, hin
        elif self.outputType == 'logic':
            result = self.__phi(oin), oin, hout, hin
        elif self.outputType == 'softmax':
            result = np.exp(oin)/np.sum(np.exp(oin)), oin, hout, hin
        else:
            raise Exception('[mlp][fwd] outputType not valid')

        if training:
            return result
        else:
            return result[0]

    def train(self, eta=0.1, beta=None, nbIte=100, momentum=0.9):
        '''
        Training using back-propagation
            :param eta: learning rate for the hidden layer
            :param beta: learning rate for the output layer
            :param nbIte: number of iterations
            :param momentum: update inertia. If no momentum is required it should be equal to 0.
        '''
        if beta is None:
            beta = eta

        updatew1 = np.zeros(self.w1.shape)
        updatew2 = np.zeros(self.w2.shape)

        for n in range(nbIte):
            outputs, oin, hout, hin = self.predict(training=True)

            if self.outputType == 'linear':
                deltaO = (outputs - self.targets)
            elif self.outputType == 'logic':
                deltaO = (outputs - self.targets) * self.__deltaPhi(oin)
            elif self.outputType == 'softmax':
                deltaO = beta * (outputs - self.targets) * outputs * (1.0 - outputs)
            else:
                raise Exception('[mlp][train] outputType not valid')

            deltaH = np.dot(deltaO, np.transpose(self.w2[:-1,:]))  * self.__deltaPhi(hin)

            updatew1 = eta * np.dot(np.transpose(self.inputs), deltaH) + momentum * updatew1
            updatew2 = beta * np.dot(np.transpose(self.__addColumn(hout)), deltaO) + momentum * updatew2

            self.w1 -= updatew1
            self.w2 -= updatew2
        return self.predict()



if __name__ == "__main__":

    '''Logic Tests'''
    eta = 0.1
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    ANDtargets = np.array([[0], [0], [0], [1]])
    ORtargets = np.array([0, 1, 1, 1]) # second format for 1 dimensional targets
    XORtargets = np.array([[0], [1], [1], [0]]) # non linearly separable

    print "XOR"
    mlp = MLP(inputs, XORtargets, nbNodes=3)
    output = mlp.train(eta, eta, 2000)
    print "Perceptron learning rule"
    print output


    '''2D test'''
    import matplotlib.pyplot as plt

    n = 100
    sigma = 0.8
    cov = [[sigma, 0], [0, sigma]]
    c = 3
    dataA = np.random.multivariate_normal([c, -c], cov, n)
    dataB = np.random.multivariate_normal([-c, c], cov, n)
    dataC = np.random.multivariate_normal([c, c], cov, n)
    dataD = np.random.multivariate_normal([-c, -c], cov, n)

    targetA = np.repeat(np.array([[1,0,0,0]]), n, axis=0)
    targetB = np.repeat(np.array([[0,1,0,0]]), n, axis=0)
    targetC = np.repeat(np.array([[0,0,1,0]]), n, axis=0)
    targetD = np.repeat(np.array([[0,0,0,1]]), n, axis=0)

    data = np.concatenate((dataA, dataB, dataC, dataD))
    target = np.concatenate((targetA, targetB, targetC, targetD))

    normData = (data - np.mean(data, axis=0)) / np.var(data, axis=0)
    # shuffle
    #p = np.random.permutation(np.shape(data)[0])
    #data = data[p]
    #target = target[p]

    mlp = MLP(normData, target, nbNodes=2)
    c = mlp.train(nbIte=10000, eta=0.1)
    c = np.argmax(c, axis=1)

    plt.plot(data[np.where(c==0),0], data[np.where(c==0),1], 'bo')
    plt.plot(data[np.where(c==1),0], data[np.where(c==1),1], 'ro')
    plt.plot(data[np.where(c==2),0], data[np.where(c==2),1], 'ko')
    plt.plot(data[np.where(c==3),0], data[np.where(c==3),1], 'go')
    plt.show()
