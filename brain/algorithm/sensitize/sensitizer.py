import numpy as np
import random as python_random

from copy import deepcopy


from brain.common import pylog

class Learner(object):
    @pylog.functionLog
    def __init__(self,name,epoch=100):
        """Initializer
           Args: name (string): name of learning model, only support xgboost for now
        """
        self.name = name
        self.seed = 42
        self.epoch = epoch
        self.model = None
        # learning model before fitting
        if name=="xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                raise ImportError('XGBRegressor is not available')

            # learning model before fitting
            self.actor = XGBRegressor(objective='reg:linear', verbosity=1, random_state=self.seed)
            # auto tuning parameters for the learning model
            self.params = {'n_estimators': [50, 100, 200],
                           'learning_rate': [0.01, 0.1, 0.2, 0.3],
                           'max_depth': range(3, 10),
                           'colsample_bytree': [0.6, 0.8, 1.0],
                           'reg_alpha':[0.1, 1.0, 25.0, 100.0],
                           'reg_lambda':[0.1, 1.0, 25.0, 100.0],
                           'min_child_weight': [1, 5, 10],
                           'subsample': [0.6, 0.8, 1.0]}
        else:
            pylog.logger.info("Support xgboost for now, current learner {} is not supported".format(self.name))
            raise AttributeError

    @pylog.functionLog
    def run(self, X_train, y_train, X_test, y_test):
        """Implementation of training and evaluating learning model
            Args:
                X_train (numpy array): training data
                y_train (numpy arary): training label for regression
                X_test (numpy array): test data
                y_test (numpy arary): test label for regression
        """
        from sklearn.model_selection import RandomizedSearchCV,KFold
        from sklearn.metrics import mean_absolute_percentage_error
        
        scoring = "neg_mean_absolute_error"  # "neg_mean_absolute_percentage_error"
        scoring_func = mean_absolute_percentage_error
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        tuner = RandomizedSearchCV(estimator=self.actor,
                                   param_distributions=self.params,
                                   n_iter=self.epoch,
                                   scoring=scoring,
                                   cv=kfold.split(X_train, y_train),
                                   verbose=0,
                                   random_state=self.seed)
        tuner.fit(X_train, y_train.ravel())
        self.model = tuner.best_estimator_
        return scoring_func(y_test.ravel(), self.model.predict(X_test))

class Explainer(object):
    @pylog.functionLog
    def __init__(self, name='shap'):
        """Initializer, only used for explaining nonlinear learning models
            Args: name (string): name of explaining model, only support shap and explain
                  shap is a open source black box explainer.
                  explain is learner's self-explain method. For xgboost, here use total_gain.
        """
        self.name = name
    
    @pylog.functionLog
    def run(self, learner, X, params, base_x=None):
        """Implementation explaining nonlinear learning models
            Args:
                learner (model): a learning model for explaining
                X (numpy array): explaining data
                params (list): a list of parameter names
            Return:
                sensi values (numpy array)
        """
        if (self.name == "shap") or (self.name is None):
            try:
                from shap import KernelExplainer
            except ImportError:
                raise ImportError('shap.KernelExplainer is not available')
            # sample data for computing shaply values
            background = np.reshape(base_x, (1, len(base_x)))  #shap.kmeans(X, 10) if X.shape[0]>10 else shap.kmeans(X, X.shape[0])
            # use kernel explainer, which compute shaply values by kernel approximation,
            # (1) this method is slower but more accurate
            # (2) this method is more general and be applied to different learning models
            # other choices could consider tree explainer, which is designed for trees
            explainer = KernelExplainer(learner.model.predict, background)
            sensi = explainer.shap_values(X)
            sensi = np.mean(sensi, axis=0)
        elif self.name == "explain":
            # use xgboost's self-explaining method, which support :
            # * 'weight': the number of times a feature is used to split the data across all trees.
            # * 'gain': the average gain across all splits the feature is used in.
            # * 'cover': the average coverage across all splits the feature is used in.
            # * 'total_gain': the total gain across all splits the feature is used in.
            # * 'total_cover': the total coverage across all splits the feature is used in.
            # use total_gain since it considers overall gain
            sensi = learner.model.get_booster().get_score(importance_type="total_gain")
            if isinstance(sensi, dict):
                # when use explain method, the output could be dict rather than a numpy array
                sensi_dict = {}
                for k in range(len(params)):
                    if 'f' + str(k) in list(sensi.keys()):
                        sensi_dict[k] = sensi['f' + str(k)]
                    else:
                        sensi_dict[k] = 0.0
                sensi = np.array(list(sensi_dict.values()))
            elif not isinstance(sensi, np.ndarray):
                pylog.logger.info("Support dict and numpy ndarray as the output format for sensi scores")
                raise TypeError
        else:
            pylog.logger.info("Support shap and explain, current explainer {} is not supported".format(self.name))
            raise AttributeError

        return sensi.squeeze()

class Analyzer(object):
    @pylog.functionLog
    def __init__(self,
                 params,
                 seed=None,
                 use_gp=False,
                 use_lasso=False,
                 use_univariate=True,
                 use_shap=True,
                 learner_name='xgboost',
                 explainer_name='shap',
                 epoch=100):
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
        self.use_gp = use_gp
        self.use_lasso = use_lasso
        self.use_univariate = use_univariate
        self.use_shap = use_shap
        self.learner_name = learner_name
        self.explainer_name = explainer_name
        self.epoch = epoch
        self.sensi = {}
        self.learner_performance = {}
        if self.use_shap:
            # use nonlinear interpreter
            # shap needs a reference point as the baseline for explanation, initialize here
            self.base_x = np.zeros(len(params))

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
    def explain_gp(self, X_train, y_train, X_test, y_test):
        """use iTuned methods for sensitization
        Args:
            X_train (numpy array): training data
            y_train (numpy arary): training label for regression
            X_test (numpy array): test data
            y_test (numpy arary): test label for regression
        Return:
            normalized sensitivity scores
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.metrics import mean_absolute_percentage_error
        model = GaussianProcessRegressor(random_state=0,normalize_y=True).fit(X_train, y_train)
        y_true, y_pred = y_test.ravel(), model.predict(X_test)
        performance = mean_absolute_percentage_error(y_true, y_pred)

        m = X_train.shape[1]
        var_y = np.std(y_train)
        var_x = np.ones(m)
        for i in range(m):
            vs = np.unique(X_train[:,i])
            y_m = []
            for j in vs:
                X_copy = deepcopy(X_train)
                X_copy[:,i] = j
                y_m.append(np.mean(model.predict(X_copy)))
            var_x[i] = np.std(y_m)

        sensi = np.array(var_x) / var_y
        sensi = self.remove_null(sensi)
        sensi = sensi / np.sum(np.abs(sensi))
        return performance, sensi
    
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
        from sklearn.metrics import mean_absolute_percentage_error
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
    def explain_univariate(self, X, y):
        """Implement univariate explainer
        Args:
            X (numpy array): training data
            y (numpy arary): training label for regression
        Return:
            normalized sensitivity scores
        """
        # use mutual info regression since it can capture pair-wise linear and nonlinear relation
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        score_func = mutual_info_regression
        model = SelectKBest(score_func, k="all")
        model.fit(X, y)
        sensi = self.remove_null(model.scores_)
        return sensi

    @pylog.functionLog
    def explain_nonlinear(self, X_train, y_train, X_test, y_test):
        """Implement nonlinear explainer
        Args:
            X_train (numpy array): training data
            y_train (numpy arary): training label for regression
            X_test (numpy array): test data
            y_test (numpy arary): test label for regression
        Return:
            normalized sensitivity scores
        """
        np.random.seed(42)
        python_random.seed(42)
        # build learning model with TRAINING data
        learner = Learner(name=self.learner_name, epoch=self.epoch)
        performance = learner.run(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        # build explaining model with ALL data
        explainer = Explainer(name=self.explainer_name)
        sensi = explainer.run(learner=learner,
                              X=np.concatenate([X_train, X_test], axis=0),
                              params=self.params,
                              base_x=self.base_x)
        return performance, self.remove_null(sensi)

    @pylog.functionLog
    def run(self, X, y):
        """Implement explaining method which ensemble lasso, univariate, and shap explainers
        Args:
            X (numpy array): all collected data
            y (numpy arary): all label for regression

        Return:
            normalized sensitivity scores
        """
        from sklearn.model_selection import train_test_split
        # split training and testing data for building learning and explaining models
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=self.seed)

        if self.use_gp:
            self.learner_performance['gp'], self.sensi['gp'] = self.explain_gp(X_train, y_train, X_test, y_test)

        if self.use_lasso:
            # use linear interpreter (default with Lasso)
            self.learner_performance['lasso'], self.sensi['lasso'] = self.explain_lasso(X_train, y_train, X_test, y_test)

        if self.use_shap:
            # use nonlinear interpreter
            # shap needs a reference point as the baseline for explanation
            self.base_x = X[0]
            self.learner_performance['shap'], self.sensi['shap'] = self.explain_nonlinear(X_train, y_train, X_test, y_test)

            if self.use_univariate:
                # use univariate values as gates, taking values from 1.0 to 2.0
                # most univariate values are very small, need to scale up (by 2 for now), otherwise tanh(x) will be too small
                self.sensi['univariate'] = self.explain_univariate(X_train, y_train)
                sensi_scaler = np.tanh(2.0 * self.sensi['univariate']) + 1.0
            else:
                sensi_scaler = 1.0

            sensi = self.sensi['shap'] * sensi_scaler
            sensi = sensi / np.sum(np.abs(sensi))
            self.sensi['aggregated'] = sensi.squeeze()
            self.sensi['shap'] = self.sensi['shap'] / np.sum(np.abs(self.sensi['shap']))

        if self.use_univariate:
            # use univariate interpreter
            self.sensi['univariate'] = self.explain_univariate(X_train, y_train)
            self.sensi['univariate'] = np.abs(self.sensi['univariate']) / np.sum(np.abs(self.sensi['univariate']))

        if len(self.sensi.keys()) <= 0:
            pylog.logger.info("Support gp, univariate, lasso, shap, none is selected")
            raise AttributeError
