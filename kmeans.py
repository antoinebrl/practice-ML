# Author : Antoine Broyelle
# Licence : MIT
import numpy as np
from utils.distances import euclidianDist

class Kmeans:

    def __init__(self, data, k=1, distance=euclidianDist):
        '''
        :param data: Training data set with samples as row elements
        :param k: number of cluster
        :param distance: metric measurement of the space. default : euclidianDist
            for k-means use euclidianDist, for k-median use manhattanDist, for k-medoids use any
            metric function which return d[i,j] the distance between point i and center j
        '''
        if k < 1:
            raise Exception("[kmeans][init] k must be greater than zero")
        self.data = data
        self.k = k
        self.distance = distance

    def clustering(self, data):
        '''
        Find the cluster each data point belongs to
            :param data: set of data point as row vector
            :return: ID of the cluster for each data point
        '''
        #distances = np.sum((data - self.centers[:, np.newaxis]) ** 2, axis=2)
        return np.argmin(self.distance(data, self.centers), axis=1)


    def train(self, nbIte=100):
        '''
        Lloyd's algorithm with pure numpy.
            :param nbIte: Maximum number of iterations if convergence is not reached before
        '''
        def __hasConverged(c1, c2):
            '''Convergence criterion. Find similarities between two set of centers'''
            return set([tuple(x) for x in c1]) == set([tuple(x) for x in c2])

        def __update(data, clusters, K):
            '''Update rule to find better centers'''
            return np.array([np.mean(data[clusters == k], axis=0) for k in range(K)])

        # Random init with data point
        select = np.random.choice(np.shape(self.data)[0], self.k, replace= False)
        self.centers = self.data[select]
        select = np.random.choice(np.shape(self.data)[0], self.k, replace= False)
        newCenters = self.data[select]

        t = 0
        while not __hasConverged(self.centers, newCenters) and t < nbIte:
            self.centers = newCenters
            clusters = self.clustering(self.data)
            newCenters = __update(self.data, clusters, self.k)
            t += 1



if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    dataA = np.random.multivariate_normal([-2, 0, 1], [[1,0,0],[0,1,0],[0,0,1]], 100)
    dataB = np.random.multivariate_normal([+3, 0, 2], [[1,0,0],[0,1,0],[0,0,1]], 100)
    dataC = np.random.multivariate_normal([0, +2, 0], [[1,0,0],[0,1,0],[0,0,1]], 100)
    data = np.concatenate((dataA, dataB, dataC))
    # shuffle
    p = np.random.permutation(np.shape(data)[0])
    data = data[p]

    km = Kmeans(data, k=3)
    km.train()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=km.clustering(data))
    ax.scatter(km.centers[:, 0], km.centers[:, 1], km.centers[:, 2], c='r', s=100)
    plt.show()


