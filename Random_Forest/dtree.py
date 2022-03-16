import numpy as np


def gini(y):
    "Return the gini impurity score for values in y"
    dicti = {}
    for i in y:
        dicti[i] = dicti.get(i, 0) + 1
    l = len(y)
    prob_sqr = 0
    for key in dicti:
        prob_sqr += (dicti[key]/l)**2
    return 1- prob_sqr


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)

    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        else:
            return self.rchild.leaf(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.y = y
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction

    def leaf(self, x_test):
        return self


class DecisionTree621:
    def __init__(self, min_samples_leaf, loss, max_features,create_leaf):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.std or gini
        self.max_features = max_features
        self.create_leaf = create_leaf

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)

        col, split = self.bestsplit(X, y, self.loss, self.max_features)
        if col == -1:
            return self.create_leaf(y)

        lchild = self.fit_(X[X[:, col] <= split], y[X[:, col] <= split])
        rchild = self.fit_(X[X[:, col] > split], y[X[:, col] > split])
        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        lst = []

        for i in X_test:
            lst.append(self.root.predict(i))
        return np.array(lst)


    def bestsplit(self, X, y, loss, max_features):
        best = {'col': -1, 'split': -1, 'loss' : loss(y)}
        size_ = min(11, X.shape[0])
        cols = np.random.choice(range(X.shape[1]), size=round(max_features*X.shape[1]), replace=False)
        for col in cols:
            candidates = np.random.choice(X[:, col], size=size_, replace=False)
            for split in candidates:
                yl = y[X[:, col] <= split]
                yr = y[X[:, col] > split]
                if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
                    continue
                l = (len(yl) * loss(yl) + len(yr) * loss(yr)) / len(y)
                if l == 0:
                    return col, split
                if l < best['loss']:
                    best = {'col': col, 'split': split, 'loss' : l}

        return best['col'], best['split']

    #
    # def create_leaf(self, y):
    #     """
    #     Return a new LeafNode for classification, passing y and mode(y) to
    #     the LeafNode constructor.
    #     """
    #     vals, counts = np.unique(y, return_counts=True)
    #     index = np.argmax(counts)
    #
    #     return LeafNode(y,vals[index])

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        residue = sum((y_pred - y_test)**2)
        total = sum((y_test - np.mean(y_test))**2)
        #r_square = r2_score(y_test, y_pred)
        r_square = 1- (residue/total)
        return r_square

    # def create_leaf(self, y):
    #     """
    # Return a new LeafNode for regression, passing y and mean(y) to
    # the LeafNode constructor.
    # """
    #     return LeafNode(y, np.mean(y))

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        #return accuracy_score(y_test, y_pred)
        return sum(np.array(y_pred) == np.array(y_test))/len(y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        vals, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return LeafNode(y,vals[index] )




