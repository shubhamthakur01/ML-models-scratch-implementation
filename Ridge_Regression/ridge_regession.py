import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def loss_gradient(X, y, B, lmbda):
    return -np.transpose(X)@(y - X@B)

def loss_ridge(X, y, B, lmbda):
    print('ridge wale')
    print(X.shape, y.shape, lmbda)
    return np.dot((y - X@B), (y-X@B)) + lmbda*(np.dot(B,B))

def loss_gradient_ridge(X, y, B, lmbda):
    return -np.transpose(X)@(y - X@B) + lmbda*B

def sigmoid(z):
    return  1 / (1 + np.exp(-z))     

def log_likelihood(X, y, B,lmbda):
    val = (y@X)@B - np.sum(np.log(1 + np.exp(X@B) ))
    return val 
    

def log_likelihood_gradient(X, y, B, lmbda):
    return -np.transpose(X)@(y-sigmoid(X@B))


def minimize(X, y, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=1e-9):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:
        X=np.column_stack((np.ones(X.shape[0]), X))
        p = p+1
        

    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)

    prev_B = B
    eps = 1e-5 # prevent division by 0
    grad =loss_gradient(X, y, B, lmbda)
    h = np.zeros(p).reshape(-1,1)
    itr = 0
    while np.linalg.norm(grad) >= precision and itr < max_iter:
        itr += 1
        grad = loss_gradient(X, y, B, lmbda)
        h += np.multiply(grad, grad)
        B = prev_B - eta*(np.divide(grad, (np.sqrt(h) + eps)) ) 
        prev_B = B
    
    return prev_B

class LinearRegression621: # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class LogisticRegression621: # REQUIRED
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter
        
    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return sigmoid(np.dot(X, self.B))
        

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        prob = self.predict_proba(X)
        return np.where(prob > 0.5 ,1,0)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621: # REQUIRED
    "Use the above classes as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        B_ = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter,addB0=False)
        B0=np.mean(y)
        self.B= np.vstack([B0,B_])


