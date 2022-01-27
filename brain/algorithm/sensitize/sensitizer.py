import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from brain.common import pylog


class Analyzer(object):
    @pylog.logit
    def __init__(self,
                 params,
                 seed=None,
                 learner=True):
        """Initializer
        Args:
             params (list): a list of parameter names
             seed (int): random seed
             learner (string): name of learner, use linear for now
        """
        self.params = params
        self.seed = seed if seed is not None else 42
        self.learner = learner

    @staticmethod
    def normalize_scores(scores):
        """normalize sensitivity scores
        Args:
            scores (numpy array): raw scores with positive and negative values
        Return:
            normalized scores
        """
        epsilon = 1e-7
        scores[np.isnan(scores)] = epsilon
        scores = np.where(np.abs(scores) > epsilon, scores, epsilon)
        return scores / np.abs(scores).sum()

    @pylog.logit
    def explain_linear(self, X_train, y_train, X_test, y_test):
        """Implement linear explainer
        Args:
            X_train (numpy array): training data
            y_train (numpy arary): training label for regression
            X_test (numpy array): test data
            y_test (numpy arary): test label for regression
        Return:
            normalized sensitivity scores
        """

        param_grid = {"alpha": np.linspace(
            0.05, 0.95, 10), "l1_ratio": [.1, .5, .7, .9, .95, .99, 1]}
        # "neg_mean_absolute_percentage_error", "neg_mean_squared_error"
        score_func = "neg_mean_absolute_error"
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        tuner = GridSearchCV(estimator=ElasticNet(),
                             param_grid=param_grid,
                             scoring=score_func,
                             cv=kfold,
                             verbose=0)
        tuner.fit(X_train, y_train.ravel())
        model = tuner.best_estimator_
        y_true, y_pred = y_test.ravel(), model.predict(X_test)
        self.performance_linear = mean_absolute_percentage_error(
            y_true, y_pred)
        return self.normalize_scores(model.coef_)

    @pylog.logit
    def run(self, X, y):
        """Implement explaining method with linear models and their coefficients as weights
        Args:
            X (numpy array): all collected data
            y (numpy arary): all label for regression

        Return:
            normalized sensitivity scores
        """

        # split training and testing data for building learning and explaining models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=self.seed)
        self.sensitivity_set = {}

        if self.learner == "linear":
            # use linear interpreter
            self.sensitivity_set['linear'] = self.explain_linear(
                X_train, y_train, X_test, y_test)
        else:
            raise AttributeError

        if len(self.sensitivity_set.values()) <= 0:
            raise AttributeError
        else:
            # aggregate multiple sensitivity scores from different explainer
            scores = np.vstack(list(self.sensitivity_set.values()))
            self.sensitivity = np.abs(scores).mean(axis=0)
            self.sensitivity = self.sensitivity / self.sensitivity.sum()
