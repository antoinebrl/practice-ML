# Author : Antoine Broyelle
# Licence : MIT
# inspired by : KTH - DD2431 Machine Learning
# https://www.kth.se/student/kurser/kurs/DD2431?l=en

import numpy as np
import math

class Node:
    def __init__(self, attr, child=None):
        self.attr = attr
        self.child = {} if child == None else child

    def addChild(self, value, node):
        self.child[value] = node

    def __repr__(self):
        return repr(self.__dict__)

class Leaf:
    def __init__(self, value):
        '''a label choice'''
        self.value = value

    def __repr__(self):
        return str(self.value)


class Dtree:
    '''Decision tree'''

    def histogram(self, data):
        ''':return: elements present in data and their distribution'''
        values, freq = np.unique(data, return_counts=True)
        freq = freq.astype(np.float)
        freq /= np.sum(freq)
        return values, freq

    def mostCommon(self, data):
        ''':return: value of the most common element'''
        values, freq = self.histogram(data)
        return values[np.argmax(freq)]

    def entropy(self, data):
        ''':return: entropy of the data set'''
        values, counts = self.histogram(data)
        clog2 = np.vectorize(lambda x : x * math.log(x,2))
        p = clog2(counts)
        return - np.sum(p)

    def gain(self, input, attr, target):
        '''
        Information gain if we look at the component attr of the data set
        :param input: data set as a matrix. Samples as row elements
        :param attr: The index of the attribute to focus on
        :param target: targeted labels
        :return: entropy reduction : Ent(S) - sum{v in values(attr)}{p(v)*Ent(S|attr=v)}
        '''
        gain = self.entropy(target)
        attrValues, freq = self.histogram(input[:, attr])
        for v, p in zip(attrValues, freq):
            gain -= p * self.entropy(target[input[:, attr] == v])
        return gain

    def bestGain(self, input, mask, target):
        '''Identify the most interesting attribute
        :param mask: mask[i] == True is the i-th attribute is considered as already used
        :return: index of the most interesting attributes
        '''
        gains = [self.gain(input, i, target) if not mask[i] else 0.0 for i in range(input.shape[1])]
        return np.argmax(gains)

    def train(self, data, target):
        '''Training algorithm is based on ID3 Heuristics'''
        def buildTree(data, target, mask):
            '''
            :param mask: mask[i] == True is the i-th attribute is considered as already used
            :return: the generated (sub)tree
            '''
            if data is None or data.ndim == 0:
                return Leaf(self.defaultTarget)
            if mask.all():
                return Leaf(self.mostCommon(target))
            if np.unique(target).shape[0] == 1:
                return Leaf(self.mostCommon(target))

            att = self.bestGain(data, mask, target)
            newMask = np.copy(mask)
            newMask[att] = True

            values = np.unique(data[:,att])
            nd = Node(attr=att)
            for v in values:
                subTree = buildTree(data[data[:,att] == v], target[data[:,att] == v], newMask)
                nd.addChild(v, subTree)
            return nd

        self.defaultTarget = self.mostCommon(target)
        mask = np.full(data.shape[1], False, dtype=bool)
        self.tree = buildTree(data, target, mask)

    def predict(self, input, tree=None):
        ''':param tree: The tree to work on. Default None. If None, self.tree is used.
            :return: vector of labels for each sample of input'''
        def followTree(tree, x):
            if isinstance(tree, Leaf):
                return tree.value
            if tree is None:
                return self.defaultTarget
            if not x[tree.attr] in tree.child:
                return self.defaultTarget
            return followTree(tree.child[x[tree.attr]], x)

        if tree == None:
            tree = self.tree
        if input.ndim == 1:
            input = input[np.newaxis, :]
        return np.array([[followTree(tree, x)] for x in input])

    def eval(self, input, target, tree=None):
        '''Evaluation of the classification on the given set
        :param tree: The tree to work on. Default None. If None, self.tree is used.
        :return: ration of well classified samples'''
        output = self.predict(input=input, tree=tree)
        return np.sum(output == target).astype(np.float) / target.shape[0]

    def prune(self, validData, validTarget):
        '''Pruning algorithm to reduce complexity. The generated tree will be replaced if a
        smaller tree which give same result on the validation set is found.'''
        def replaceNode(tree):
            '''Recursive method to generate all possible trees'''
            # Todo : find better pruning strategy
            if isinstance(tree, Leaf):
                return ()
            alternatives = (Leaf(self.defaultTarget),)
            for v in tree.child :
                for subTree in replaceNode(tree.child[v]):
                    newConnection = tree.child.copy()
                    newConnection[v] = subTree
                    alternatives += (Node(tree.attr, newConnection),)
            return alternatives

        defaultValidationError = self.eval(validData, validTarget)
        allPossibleTrees = replaceNode(self.tree)
        score = np.array([self.eval(validData, validTarget, tree) for tree in allPossibleTrees])
        score[score < defaultValidationError] = 0
        bestTreeIndex = np.argmax(score)
        if score[bestTreeIndex] != 0:
            self.tree = allPossibleTrees[bestTreeIndex]

if __name__ == "__main__":
    data = np.array([[1,2],[1,2],[2,1],[1,1]])
    target = np.array([[True], [True], [False], [True]])

    dt = Dtree()

    data = np.array([[1, 1, 1, 1, 3, 1],[1, 1, 1, 1, 3, 2],[1, 1, 1, 3, 2, 1],[1, 1, 1, 3, 3, 2],
     [1, 1, 2, 1, 2, 1],[1, 1, 2, 1, 2, 2],[1, 1, 2, 2, 3, 1],[1, 1, 2, 2, 4, 1],[1, 1, 2, 3, 1, 2],
     [1, 2, 1, 1, 1, 2],[1, 2, 1, 1, 2, 1],[1, 2, 1, 1, 3, 1],[1, 2, 1, 1, 4, 2],[1, 2, 1, 2, 1, 1],
     [1, 2, 1, 2, 3, 1],[1, 2, 1, 2, 3, 2],[1, 2, 1, 2, 4, 2],[1, 2, 1, 3, 2, 1],[1, 2, 1, 3, 4, 2],
     [1, 2, 2, 1, 2, 2],[1, 2, 2, 2, 3, 2],[1, 2, 2, 2, 4, 1],[1, 2, 2, 2, 4, 2],[1, 2, 2, 3, 2, 2],
     [1, 2, 2, 3, 3, 1],[1, 2, 2, 3, 3, 2],[1, 3, 1, 1, 2, 1],[1, 3, 1, 1, 4, 1],[1, 3, 1, 2, 2, 1],
     [1, 3, 1, 2, 4, 1],[1, 3, 1, 3, 1, 2],[1, 3, 1, 3, 2, 2],[1, 3, 1, 3, 3, 1],[1, 3, 1, 3, 4, 1],
     [1, 3, 1, 3, 4, 2],[1, 3, 2, 1, 2, 2],[1, 3, 2, 2, 1, 2],[1, 3, 2, 2, 2, 2],[1, 3, 2, 2, 3, 2],
     [1, 3, 2, 2, 4, 1],[1, 3, 2, 2, 4, 2],[1, 3, 2, 3, 1, 1],[1, 3, 2, 3, 2, 1],[1, 3, 2, 3, 4, 1],
     [1, 3, 2, 3, 4, 2],[2, 1, 1, 1, 3, 1],[2, 1, 1, 1, 3, 2],[2, 1, 1, 2, 1, 1],[2, 1, 1, 2, 1, 2],
     [2, 1, 1, 2, 2, 2],[2, 1, 1, 2, 3, 1],[2, 1, 1, 2, 4, 1],[2, 1, 1, 2, 4, 2],[2, 1, 1, 3, 4, 1],
     [2, 1, 2, 1, 2, 2],[2, 1, 2, 1, 3, 1],[2, 1, 2, 1, 4, 2],[2, 1, 2, 2, 3, 1],[2, 1, 2, 2, 4, 2],
     [2, 1, 2, 3, 2, 2],[2, 1, 2, 3, 4, 1],[2, 2, 1, 1, 2, 1],[2, 2, 1, 1, 2, 2],[2, 2, 1, 1, 3, 1],
     [2, 2, 1, 2, 3, 2],[2, 2, 1, 3, 1, 1],[2, 2, 1, 3, 1, 2],[2, 2, 1, 3, 2, 2],[2, 2, 1, 3, 3, 2],
     [2, 2, 1, 3, 4, 2],[2, 2, 2, 1, 1, 1],[2, 2, 2, 1, 3, 2],[2, 2, 2, 1, 4, 1],[2, 2, 2, 1, 4, 2],
     [2, 2, 2, 2, 2, 1],[2, 2, 2, 3, 4, 1],[2, 3, 1, 1, 1, 1],[2, 3, 1, 2, 1, 1],[2, 3, 1, 2, 3, 1],
     [2, 3, 1, 3, 1, 2],[2, 3, 1, 3, 3, 1],[2, 3, 1, 3, 4, 2],[2, 3, 2, 1, 3, 2],[2, 3, 2, 2, 1, 1],
     [2, 3, 2, 2, 1, 2],[2, 3, 2, 2, 2, 1],[2, 3, 2, 3, 3, 2],[3, 1, 1, 1, 1, 1],[3, 1, 1, 1, 1, 2],
     [3, 1, 1, 2, 1, 1],[3, 1, 1, 2, 2, 2],[3, 1, 1, 3, 2, 2],[3, 1, 2, 1, 1, 1],[3, 1, 2, 1, 2, 2],
     [3, 1, 2, 2, 2, 2],[3, 1, 2, 2, 3, 2],[3, 1, 2, 3, 2, 2],[3, 2, 1, 1, 1, 1],[3, 2, 1, 1, 4, 2],
     [3, 2, 1, 2, 1, 2],[3, 2, 1, 2, 4, 2],[3, 2, 2, 1, 1, 1],[3, 2, 2, 1, 1, 2],[3, 2, 2, 1, 3, 2],
     [3, 2, 2, 3, 1, 1],[3, 2, 2, 3, 2, 1],[3, 2, 2, 3, 4, 1],[3, 3, 1, 1, 1, 1],[3, 3, 1, 1, 2, 1],
     [3, 3, 1, 1, 4, 2],[3, 3, 1, 2, 3, 2],[3, 3, 1, 2, 4, 2],[3, 3, 1, 3, 1, 2],[3, 3, 1, 3, 2, 1],
     [3, 3, 1, 3, 2, 2],[3, 3, 1, 3, 4, 2],[3, 3, 2, 1, 1, 1],[3, 3, 2, 1, 3, 2],[3, 3, 2, 1, 4, 1],
     [3, 3, 2, 1, 4, 2],[3, 3, 2, 3, 1, 2],[3, 3, 2, 3, 2, 2],[3, 3, 2, 3, 3, 2],[3, 3, 2, 3, 4, 2]])

    target = np.array([[True],[True],[True],[True],[True],[True],[True],[True],[True],[True],
     [False],[False],[False],[True],[False],[False],[False],[False],[False],[False],[False],[False],
     [False],[False],[False],[False],[False],[False],[False],[False],[True],[False],[False],[False],
     [False],[False],[True],[False],[False],[False],[False],[True],[False],[False],[False],[False],
     [False],[True],[True],[False],[False],[False],[False],[False],[False],[False],[False],[False],
     [False],[False],[False],[True],[True],[True],[True],[True],[True],[True],[True],[True],[True],
     [True],[True],[True],[True],[True],[True],[True],[False],[True],[False],[False],[False],[True],
     [True],[False],[False],[True],[True],[True],[False],[False],[True],[False],[False],[False],
     [False],[True],[False],[True],[False],[True],[True],[False],[True],[False],[False],[True],
     [True],[True],[True],[True],[True],[True],[True],[True],[True],[True],[True],[True],[True],
     [True],[True],[True]])

    dt.train(data,target)
    values, freq = np.unique(target - dt.predict(data), return_counts=True)
    print values
    print freq
    print dt.tree

    t = target[0:20]
    t[0] = [False]
    print dt.eval(data[0:20], target[0:20])

    dt.prune(data[0:10], target[0:10])
    print dt.tree
    print dt.eval(data[0:20], target[0:20])
