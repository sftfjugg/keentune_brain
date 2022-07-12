import os
import pickle
import numpy as np

from time import sleep
from datetime import datetime

from brain.algorithm.sensitize.sensitizer import Analyzer

from brain.common import tools
from brain.common import pylog
from brain.common.dataset import DataSet
from brain.common.config import Config

@pylog.functionLog
def _parseSenstizeResult(sensitize_results, knobs):
    """Transfor sensitize_weight dict to response result

    Args:
        sensitize_results (tuple): (a list of sorted parameter names, a list of sorted weights, confidence)
        knobs (list): a knob list selected by sensitivity
    """
    (params, weights, _) = sensitize_results
    knobs_selected = []
    sorted_indice = []
    for knob in knobs:
        if knob['name'] not in params:
            continue
        index = params.index(knob['name'])
        sorted_indice.append(index)
        knob['weight'] = weights[index]
        knobs_selected.append(knob)

    return [knobs_selected[sorted_indice.index(i)] for i in range(len(params))]


@pylog.functionLog
def _sensitizeSelect(sensitize_weight, topN=10, confidence_threshold=0.9):
    """Recommend sensitivity results

        Args:
            sensitize_weight (dict): a dict of parameter and weights, e.g., {"param name": weight }
            topN (int): a predefined number of parameters to recommend
            confidence_threshold (float): a predefined confidence interval for recommending parameters

        Return: a tuple (params_sorted, weights_sorted, confidence_interal)
            params_sorted (list): a sorted list of recommended parameters,
            weights_sorted (list): a sorted list of the sensitivity values of recommended parameters,
            confidence_interal (float): a scaler showing the confidence of recommended parameters
    """
    # check topN input value
    if topN is None or topN <= 0:
        topN = 10

    # check confidence_threshold value
    if confidence_threshold is None or confidence_threshold <= 0.0 or confidence_threshold >= 1.0:
        confidence_threshold = 0.9

    params = []
    weights = []
    # extract parameter names and weights
    for s in sensitize_weight.items():
        params.append(s[0])
        weights.append(s[1])

    # sort weights descendingly
    sorted_indice = np.argsort(weights)[::-1]
    params_sorted = [params[i] for i in sorted_indice]
    # use absolute values to sort
    weights_sorted = np.abs([weights[i] for i in sorted_indice])

    weights_cumsum = np.cumsum(np.array(weights_sorted))
    try:
        index = np.where(weights_cumsum >= confidence_threshold)[0][0]
    except IndexError:
        index = 0
    k = topN if topN <= index else index + 1
    confidence = weights_cumsum[k - 1]
    # use original values to output
    weights_sorted = np.array([weights[i] for i in sorted_indice])

    """ 
    Return complete knobs list with weights to user, filter knobs by themselves.
    Author: runzhe    Date: 2022.6.1
    """
    return (params_sorted, weights_sorted, confidence)
    # return (params_sorted[:k], weights_sorted[:k], confidence)



@pylog.functionLog
def _loadData(sensitize_data):
    """Load and prepare data for analyzing

    Args:
        sensitize_data (DataSet): dataset to sensitize
    Return:
        X (numpy array of shape (n,m)): training data,
        y (numpy array of shape (n,)): training target for regression,
        params (list): a list of parameter names
    """

    # extract parameter names
    knobs = np.load(sensitize_data.knobs_path, allow_pickle=True)
    params = [k['name'] for k in knobs]

    # load data
    X = np.load(sensitize_data.points_path, allow_pickle=True)
    X = tools.normalizePts(X, knobs)
    
    # load score
    bench = np.load(sensitize_data.bench_path, allow_pickle=True)

    # find benchmark results with highest weight as the main objective for analysis
    bench_index = np.argsort([bench[k]['weight'] for k in bench.keys()])[-1]
    score = np.load(sensitize_data.score_path, allow_pickle=True)
    y = score[:, bench_index]
    return X, y, params


@pylog.functionLog
def _getStability(Z):
    """Compute stability according

    Args:
        mask (numpy array): A mask matrix with binary values for selected parameters.

    Return:
        stable: The stability of the feature selection procedure
    """
    n_trails, n_params = Z.shape
    # get empirical mean factor
    empirical_factor = float(n_trails) / (n_trails - 1)

    # get mean of mask values and compute 1st step stable score
    Z_mean = Z.mean(axis=0)
    stable = np.multiply(Z_mean, 1 - Z_mean).mean()
    # get expected value of mask as normalizing facotor
    mask_expected = Z_mean.sum()
    normalize_factor = (mask_expected / n_params) * (1 - mask_expected / n_params)

    # compute stable scores
    stable = empirical_factor * stable / normalize_factor
    stable = 1 - stable
    if np.isnan(stable):
        stable = 1.0
    return stable


@pylog.functionLog
def _computeStability(scores, params):
    """Compute two types of stability scores

    Args:
        log (dict): a dict of sensitivity results from different trials,
             e.g., log = {0: {"sensitivity": 0.01}, 1: {"sensitivity": 0.02}, ....}
        params (list): a list of parameter names
        plot (bool): a flag for ploting stability results and orders of parameters


    Return:
        results: a dict of stability results, e.g.,
                 results = {"stable_I": np.zeros(shape=(number of parameters, )),
                            "stable_II": np.zeros(shape=(number of parameters, ))}
    """
    n_trials, n_params = scores.shape
    # find sorted indice of params
    params_order = np.zeros(shape=(n_trials, n_params))
    for k in range(n_trials):
        params_sorted = [params[i] for i in np.abs(scores[k]).argsort()[::-1]]
        for j, p in enumerate(params):
            params_order[k, j] = params_sorted.index(p)

    # compute stability for different ratios of selected parameters
    ratios = list(np.round(np.arange(0.1, 1.0, 0.1), 2)) + [0.99]
    stable_types = ['stable_I', 'stable_II']
    results = {}
    for t in stable_types:
        results[t] = np.zeros(len(ratios))

    Z = np.zeros((len(ratios), n_trials, n_params), dtype=np.int8)
    for i, r in enumerate(ratios):
        results[r] = {}
        # get number of selected parameters
        n = int(np.ceil(n_params * r))
        for k in range(n_trials):
            # find indice of selected parameters
            topK = np.where(params_order[k] < n)
            Z[i, k, topK] = 1
            if k == 0:
                selected_params = set([params[j] for j in topK[0]])
            else:
                selected_params = selected_params.intersection(set([params[j] for j in topK[0]]))

        results['stable_I'][i] = _getStability(Z[i]) if n_trials > 1 else 1.0
        results['stable_II'][i] = float(len(selected_params)) / n

    # directories to save stable scores and parameter orders box-plot
    dump_folder_path = os.path.join(
        Config.SENSI_DATA_PATH, 
        datetime.now().strftime("%y-%m-%d-%H-%M-%S"))


    if not os.path.exists(dump_folder_path):
        os.makedirs(dump_folder_path)

    params_name_file = os.path.join(dump_folder_path, "name.pkl")
    params_order_file = os.path.join(dump_folder_path, "order.pkl")

    params_order_median = np.median(params_order, axis=0)
    params_order_median_indice = params_order_median.argsort()
    params_order_sorted = [params[i] for i in params_order_median_indice]

    pickle.dump(params_order_sorted, open(params_name_file, 'wb+'))
    pickle.dump(params_order[:, params_order_median_indice], open(params_order_file, 'wb+'))


@pylog.functionLog
def _sensitizeImpl(sensitize_data, explainer='shap', trials=0, epoch=50, topN=10, threshold=0.9):
    """Implementation of sensitive parameter identification algorithm

    Args:
        data_path (string): path of numpy data

    Return:
        sensitize_result: a dict of parameters sensitivity scores, keys are sorted descendingly according to the scores
                          e.g., sensitize_result = {"parameter_name": float value}
    """
    X, y, params = _loadData(sensitize_data)
    sensitize_result, sensi_file = _sensitizeRun(X=X, y=y, params=params, 
                                    learner="xgboost", explainer=explainer, 
                                    epoch=epoch, trials=trials)
    sensitize_result = _sensitizeSelect(sensitize_weight=sensitize_result, 
                                        topN=topN,
                                        confidence_threshold=threshold)

    sleep(10)
    return True, _parseSenstizeResult(sensitize_result, sensitize_data.knobs), sensi_file


@pylog.functionLog
def _sensitizeRun(X, y, params, learner="xgboost", explainer="shap", epoch=50, trials=0, verbose=1):
    """Implementation of sensitive parameter identification algorithm

    Args:
        X (numpy array of shape (n,m)): training data
        y (numpy array of shape (n,)): training label
        params (list of strings): a list of parameter names
        learner (string) : name of learner, only xgboost for now
        trials (int): number of trails to run for get average sensitivity scores from multiple trials.

    Return:
        sensitize_result: a dict of parameters sensitivity scores, keys are sorted descendingly according to the scores
                              e.g., sensitize_result = {"parameter_name": float value}
        """

    # set seed for reproducible results
    seed = 42
    np.random.seed(seed)
    if trials > 1:
        seeds = [int(i) for i in list(np.random.randint(100, size=trials))]
    else:
        trials = 1
        seeds = [seed]

    # use_xx flag controls the use of specific explaining algorithms (gp, lasso, univariate, shap).
    # To use multiple explaining algorithms at the same time, edit the flags here.
    # Future consider control these flags through sensitize interface
    use_gp = True if explainer=='gp' else False
    use_lasso = True if explainer=='lasso' else False
    (use_shap, use_univariate) = (True, True) if (explainer=='shap') or (explainer=='explain') else (False, False)
    use_univariate = True if (explainer=='univariate') or (use_univariate==True) else False
    
    # if none specified, use shap as default
    if not (use_gp or use_lasso or use_univariate or use_shap):
        use_shap, use_univariate = True, True
        learner, explainer = "xgboost", "shap"

    if epoch is None or epoch <= 1:
        epoch = 50


    log = {}
    sensi = np.zeros((trials, len(params)))
    log['learner'] = learner
    log['explainer'] = explainer
    log['parameters'] = params
    
    for i, s in enumerate(seeds):
        # initialize sensitier
        sensitizer = Analyzer(params=params,
                              seed=s,
                              use_gp=use_gp,
                              use_lasso=use_lasso,
                              use_univariate=use_univariate,
                              use_shap=use_shap,
                              learner_name=learner,
                              explainer_name=explainer,
                              epoch=epoch)
        # run sensitizer with collected data
        sensitizer.run(X, y)
        if verbose > 0:
            log[i] = {}
            log[i]['seed'] = s
            for k in ['gp','lasso','univariate','shap','aggregated']:
                if k in sensitizer.learner_performance.keys():
                    log[i]['{}_performance'.format(k)] = sensitizer.learner_performance[k]
                    pylog.logger.info("trial:{}, {} performance: {}".format(i, k, sensitizer.learner_performance[k]))
                    print("trial:{}, {} performance: {}".format(i, k, sensitizer.learner_performance[k]))
                if k in sensitizer.sensi.keys():
                    log[i]['{}_sensitivity'] = sensitizer.sensi[k]
                    pylog.logger.info("trial:{}, {} sensitivity: {}".format(i, k, sensitizer.sensi[k]))
                    print("trial:{}, {} sensitivity: {}".format(i, k, sensitizer.sensi[k]))

        if explainer not in ['gp','lasso','univariate']:
            sensi[i] = sensitizer.sensi['aggregated']
        else:
            sensi[i] = sensitizer.sensi[explainer]

    dump_folder_path = os.path.join(
        Config.SENSI_DATA_PATH, 
        datetime.now().strftime("%y-%m-%d-%H-%M-%S"))

    if not os.path.exists(dump_folder_path):
        os.makedirs(dump_folder_path)
    sensi_file = os.path.join(dump_folder_path, "sensi.pkl")
    pickle.dump(sensi, open(sensi_file, 'wb+'))

    if verbose > 0:
        _computeStability(scores=sensi, params=params)

    # compute median of sensitivity scores
    sensi_mean = np.mean(sensi, axis=0)
    sensi_mean = sensi_mean / np.abs(sensi_mean).sum()
    # sort sensitivity scores in descending order
    result = {}
    for i in range(len(params)):
        result[params[i]] = sensi_mean[i]
    # return sorted(sensitize_result.items(), key=lambda d: d[1], reverse=True)
    return result, sensi_file


@pylog.functionLog
def sensitize(data_name="", explainer='shap', trials=0, epoch=50, topN=10, threshold=0.9):
    """Call sensitivity estimation algorithms

        Args:
            data_name (str): path to prepared data for sensitivity estimation
            explainer (str): name of the explaining algorithm, supporting four methods: gp, lasso, univariate, shap
            trials (int): a number for running sensitize to get stable results
            epoch (int): a predefined number for training learning models
            topN (int): a predefined number for recommending parameters
            threshold (float): a predefined confidence interval for recommending parameters

        Return: 
            suc (bool): indicate if run success
            sensitize_result (dict): a dictionary containing sensitivity scores for recommended parameters
            sensi_file: the path to detialed sensi data from all trials
    """

    try:
        sensitize_data = DataSet(data_name)
    
    except Exception as e:
        return False, "Can not load data to sensitize: {}".format(e)
    
    else:
        suc, sensitize_result, sensi_file = _sensitizeImpl(sensitize_data, explainer, trials, epoch, topN, threshold)
        if not suc:
            return False, "Get sensitive parameter failed: {}".format(sensitize_result), ""

        return True, sensitize_result, sensi_file