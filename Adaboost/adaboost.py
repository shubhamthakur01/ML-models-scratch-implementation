import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pandas as pd

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    ### BEGIN SOLUTION
    df = pd.read_csv(filename, header = None)
    #X = df.loc[:, :56].values
    X = df.values[:, 0:- 1]
    Y = df.values[:, -1]
    Y[Y ==0] = -1
    
    ### END SOLUTION
    return X, Y.astype(float)


def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N) / N
    #w =[1/N for i in range(len(X))]
    ### BEGIN SOLUTION
    for i in range(num_iter):
        h = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        h.fit(X, y, sample_weight=d)
        trees.append(h)
        y_pred = h.predict(X)
        err = sum(d*(y_pred != y))
        alpha = np.log( (1-err)/err ) if err != 0 else 1
        d = np.array([wt if y_pred[idx] == y[idx] else wt*((1-err)/err) for idx, wt in enumerate(d) ])
        d = d/sum(d)
        trees_weights.append(alpha)
    ### END SOLUTION
    return trees, trees_weights


def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)
    ### BEGIN SOLUTION

    for tree, wgts in zip(trees, trees_weights):
        y_pred = tree.predict(X)*wgts
        y += y_pred
    ### END SOLUTION
    return np.sign(y)
