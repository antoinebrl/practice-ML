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
        self.w1 = (np.random.rand(self.dimIn + 1, self.nbNodes) - 0.5) * 2 / np.sqrt(self.dimIn)
        self.w2 = (np.random.rand(self.nbNodes + 1, self.dimOut) - 0.5) * 2 / np.sqrt(self.nbNodes)


    def __addColumn(self, inputs):
        return np.concatenate((inputs, np.ones((np.shape(inputs)[0],1))),axis=1)


    def __phi(self,x):
        '''Sigmoid function for activation'''
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def __deltaPhi(self,x):
        '''Derivative of the Sigmoid function phi'''
        return (1.0 + self.__phi(x))*(1.0 - self.__phi(x)) / 2.0


    def fwd(self, inputs=None, training=False):
        '''
        Recall/Forward step of the back-propagation algorithm
            :param inputs:
            :param training: if called with training = True, temporary calculations are returned
            :return: oout: output of the network. oout = phi(oin)
                    oin: input of output nodes. oin = hout*W2
                    hout : output of the first layer. hout = phi(hin)
                    hin : intput of the hidden nodes. hin = inputs*W1
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

    def train(self, eta=0.1, beta=None, nbIte=100):
        '''
        Training using back-propagation
            :param eta: learning rate for the hidden layer
            :param beta: learning rate for the output layer
            :param nbIte: number of iterations
        '''
        if beta is None:
            beta = eta

        for n in range(nbIte):
            outputs, oin, hout, hin = self.fwd(training=True)

            if self.outputType == 'linear':
                deltaO = (outputs - self.targets)
            elif self.outputType == 'logic':
                deltaO = (outputs - self.targets) * self.__deltaPhi(oin)
            elif self.outputType == 'softmax':
                deltaO = beta * (outputs - self.targets) * outputs * (1.0 - outputs)
            else:
                raise Exception('[mlp][train] outputType not valid')

            deltaH = np.dot(deltaO, np.transpose(self.w2[:-1,:]))  * self.__deltaPhi(hin)

            self.w1 -= eta * np.dot(np.transpose(self.inputs), deltaH)
            self.w2 -= beta * np.dot(np.transpose(self.__addColumn(hout)), deltaO)

        return self.fwd()



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

