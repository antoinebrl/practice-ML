# Author : Antoine Broyelle
# Licence : MIT
# inspired by : ENSIMAG - 4MMFDAS6 : Data mining and multivariate statistical analysis
# http://ensimag.grenoble-inp.fr/cursus-ingenieur/data-mining-and-multivariate-statistical-analysis-4mmfdas6-842620.kjsp

import numpy as np

class LinReg:
    '''Linear Regression. Least-squares optimisation'''

    def __init__(self, inputs, targets):
        '''Constructor'''
        # target is a column vector
        #if np.ndim(targets) != 2 or np.shape(targets)[1] != 1:
        #    raise Exception('[linreg][init] targets variable must be a column vector')

        self.inputs = self.__addColumn(inputs) # add bias
        self.targets = targets

        self.W = np.linalg.inv(np.dot(np.transpose(self.inputs), self.inputs))
        self.W = np.dot(np.dot(self.W, np.transpose(self.inputs)), targets)

    def __addColumn(self, inputs):
        '''Insert column with ones'''
        return np.concatenate((inputs, np.ones((np.shape(inputs)[0],1))),axis=1)

    def eval(self, inputs=None):
        if inputs is None:
            inputs = self.inputs
        else:
            inputs = self.__addColumn(inputs) # add bias
        return np.dot(inputs, self.W)


    def error(self, inputs=None, targets=None):
        '''RSS. Residual sum of squares'''
        if inputs is None or targets is None:
            inputs = self.inputs
            targets = self.targets
        else:
            inputs = self.__addColumn(inputs) # add bias

        output = np.dot(inputs, self.W)
        error = np.sum((output - targets)**2)
        return error


if __name__ == "__main__":
    inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    ORtargets = np.array([[0], [1], [1], [1]])

    linreg = LinReg(inputs, ORtargets)

    output = linreg.eval(inputs)
    print "Regression :"
    print output
    print "Error :", linreg.error()


    classification = np.where(output >= 0.5, 1, 0)
    print "Classification :"
    print classification

