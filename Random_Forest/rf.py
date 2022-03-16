from dtree import *
from sklearn.utils import resample
from sklearn.metrics import r2_score
class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        trees = [None for i in range(self.n_estimators)]
        index = [None for i in range(self.n_estimators)]
        for i in range(self.n_estimators):
            #X_ , Y_ = resample(X, y, n_samples = len(X))
            idx = resample(range(len(X)), n_samples=len(X))
            X_, Y_ = X[idx], y[idx]
            DT_object = DecisionTree621(self.min_samples_leaf, self.loss, self.max_features, self.create_leaf)
            DT_object.fit(X_, Y_)
            T_i = DT_object.root
            trees[i] = T_i
            index[i] = idx


        self.trees = trees
        self.index = index

        if self.oob_score:
             self.oob_score_ = self.oob_score_compute(X,y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        #self.trees = self.all_tree

        self.loss = np.std
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.oob_score = oob_score
        #self.create_leaf = RegressionTree621().create_leaf()

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        pred = []
        for i in range(len(X_test)):
            weighted = 0
            nobs = 0
            for tree in self.trees:
                leaf_ = tree.leaf(X_test[i,:])
                weighted += (leaf_.prediction)*(leaf_.n)
                nobs += leaf_.n
            pred.append(weighted/nobs)

        return pred

    def oob_score_compute(self, X, y):
        oob_count = np.zeros(len(X))
        oob_pred = np.zeros(len(X))
        for i in range(len(self.trees)):
            tree = self.trees[i]
            indexes = self.index[i]
            oob_index = set(range(len(X))) - set(indexes)
            for j in oob_index:
                leaf_ = tree.leaf(X[j, :])
                oob_pred[j] += (leaf_.prediction) * (leaf_.n)
                oob_count[j] += leaf_.n

        oob_avg_pred = oob_pred[oob_count>0]/oob_count[oob_count>0]
        r_square = r2_score(y[oob_count>0], oob_avg_pred)
        return r_square



    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        residue = sum((y_pred - y_test)**2)
        total = sum((y_test - np.mean(y_test))**2)
        #r_square = r2_score(y_test, y_pred)
        r_square = 1- (residue/total)
        return r_square


    def create_leaf(self, y):
        """
    Return a new LeafNode for regression, passing y and mean(y) to
    the LeafNode constructor.
    """
        return LeafNode(y, np.mean(y))




class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        #self.trees =  self.all_tree
        self.loss = gini
        self.oob_score = oob_score


    def predict(self, X_test) -> np.ndarray:
        pred = []
        for i in range(len(X_test)):
            dicti = {}
            for tree in self.trees:
                leaf_ = tree.leaf(X_test[i,:])
                leaf_vals =leaf_.y
                for label in leaf_vals:
                    dicti[label] = dicti.get(label, 0) + 1
            max_key = max(zip(dicti.values(), dicti.keys()))[1]
            pred.append(max_key)
        return np.array(pred)

    def oob_score_compute(self, X, y):
        oob_count = np.zeros(len(X))
        oob_pred = np.zeros(len(X))
        oob_dict = [{} for i in range(len(X))]
        for i in range(len(self.trees)):
            tree = self.trees[i]
            indexes = self.index[i]
            oob_index = set(range(len(X))) - set(indexes)
            for j in oob_index:
                leaf_ = tree.leaf(X[j, :])
                leaf_vals = leaf_.y
                for label in leaf_vals:
                    dicti =oob_dict[j]
                    dicti[label] = dicti.get(label, 0) + 1
                    oob_count[j] += 1

        for idx ,cnt in enumerate(oob_count):
            if cnt > 0:
                oob_pred[idx] = max(zip(oob_dict[idx].values(), oob_dict[idx].keys()))[1]

        acc = sum(np.array(oob_pred[oob_count>0]) == np.array(y[oob_count>0]))/len(y[oob_count>0])
        return acc


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return sum(np.array(y_pred) == np.array(y_test))/len(y_test)


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        vals, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return LeafNode(y,vals[index])