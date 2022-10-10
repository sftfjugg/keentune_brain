import numpy as np
import random as python_random

from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from brain.common import pylog

class Analyzer(object):
    @pylog.functionLog
    def __init__(self,
                 params,
                 seed=None):
        """Initializer
        Args:
             params (list): a list of parameter names
             seed (int): random seed
             use_lasso (bool): True for use lasso explainer, default as True
             use_univariate (bool): True for use univariate explainer, default as True
             use_shap (bool): True for use shap explainer, default as True
             learner_name (string): name of learner used for explaining
             explainer_name (string): name of explainer
        """
        self.params = params
        self.seed = seed if seed is not None else 42
        self.sensi = {}
        self.learner_performance = {}


    @staticmethod
    def remove_null(data):
        """remove null values in sensitivity scores
        Args:
            data (numpy array): raw data with positive and negative values
        Return:
            scores (numpy array)
        """
        epsilon = 1e-7
        data[np.isnan(data)] = epsilon
        data = np.where(np.abs(data)>epsilon, data, epsilon)
        return data
    
    
    @pylog.functionLog
    def explain_lasso(self, X_train, y_train, X_test, y_test):
        """Implement linear explainer with Lasso
        Args:
            X_train (numpy array): training data
            y_train (numpy arary): training label for regression
            X_test (numpy array): test data
            y_test (numpy arary): test label for regression
        Return:
            normalized sensitivity scores
        """
        from sklearn.linear_model import LassoCV, Lasso
        lassocv = LassoCV(alphas=None, cv=5, max_iter=100000, normalize=True)
        lassocv.fit(X_train, y_train)
        model = Lasso(max_iter=10000, normalize=True)
        model.set_params(alpha=lassocv.alpha_)
        model.fit(X_train, y_train)
        y_true, y_pred = y_test.ravel(), model.predict(X_test)
        performance = mean_absolute_percentage_error(y_true, y_pred)
        sensi = self.remove_null(model.coef_)
        sensi = sensi / np.sum(np.abs(sensi))
        return performance,sensi


    @pylog.functionLog
    def run(self, X, y):
        """Implement explaining method which ensemble lasso, univariate, and shap explainers
        Args:
            X (numpy array): all collected data
            y (numpy arary): all label for regression

        Return:
            normalized sensitivity scores
        """
        # split training and testing data for building learning and explaining models
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=self.seed)

        # use linear interpreter (default with Lasso)
        self.learner_performance['lasso'], self.sensi['lasso'] = self.explain_lasso(X_train, y_train, X_test, y_test)

        if len(self.sensi.keys()) <= 0:
            pylog.logger.info("Support gp, univariate, lasso, shap, none is selected")
            raise AttributeError
