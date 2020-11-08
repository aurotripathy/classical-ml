# https://geoffruddock.com/adaboost-from-scratch-in-python/
import numpy as np

from sklearn.tree import DecisionTreeClassifier     
from utils import plot_adaboost, plot_staged_adaboost, make_toy_dataset


class AdaBoost:
    """
    Adaboost ensemble classifier from scratch
    """
    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None
    
    def _check_X_y(self, X, y):
        """
        validate assumptions about format of input data
        """
        assert set(y) == {-1, 1}, "outputs must be in the rahe -1 and 1"
        return X, y


    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int):
        """ Fit the model using the training data"""
        X, y = self._check_X_y(X, y)
        n = X.shape[0]

        self.stumps = np.zeros(shape=iterations, dtype=object)
        self.stump_weights = np.zeros(shape=iterations)
        self.errors = np.zeros(shape=iterations)
        self.sample_weights = np.zeros(shape=(iterations, n))

        # at time step 0, init weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iterations):
            current_sample_weights = self.sample_weights[t]
            # fit weak learner
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=current_sample_weights)

            # calculate arror and prediction from weak learner prediction
            stump_pred = stump.predict(X)
            err = current_sample_weights[(stump_pred !=y)].sum()
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = (
                current_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )
            new_sample_weights /= new_sample_weights.sum()

            # if not final iteration, update sample weights for time-step t+1
            if t+1 < iterations:
                self.sample_weights[t+1] = new_sample_weights
            
            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self

    def predict(self, X):
        """
        Make predictions using fitted model
        """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))





X, y = make_toy_dataset(n=10, random_seed=10)
plot_adaboost(X, y, plotfile='plot1.png')

# benchamrk against an existing booster
from sklearn.ensemble import AdaBoostClassifier
bench = AdaBoostClassifier(n_estimators=10, algorithm='SAMME').fit(X, y)
plot_adaboost(X, y, bench, plotfile='plot-bench.png')

clf = AdaBoost().fit(X, y, iterations=10)
plot_adaboost(X, y, clf, plotfile='plot-scratch.png')

clf = AdaBoost().fit(X, y, iterations=10)
plot_staged_adaboost(X, y, clf, plotfile='staged-plot.png')