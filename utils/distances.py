import numpy as np

def euclidianDist(data, centers):
    '''Compute the euclidian distance of each data point to each center'''
    return np.sqrt(np.sum((data - centers[:, np.newaxis]) ** 2, axis=2)).T

def manhattanDist(data, centers):
    '''Compute the Manhattan distance of each data point to each center'''
    return np.sum(np.abs(data - centers[:, np.newaxis]), axis=2).T

