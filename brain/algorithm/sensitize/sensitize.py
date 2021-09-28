import os
import re
import pickle
import numpy as np

from time import sleep
from datetime import datetime

from brain.algorithm.sensitize.sensitizer import Analyzer

from brain.common import tools
from brain.common.config import Config
from brain.common.pylog import normalFuncLog, functionLog


@normalFuncLog
def _parseSenstizeResult(sensitize_results, knobs):
    """Transfor sensitize_weight dict to response result

    Args:
        sensitize_results (tuple): (a list of sorted parameter names, a list of sorted weights, confidence)
        knobs (list): a knob list selected by sensitivity
    """
    (params, weights, confidence) = sensitize_results
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


@normalFuncLog
def _sensitizeSelect(sensitize_weight, topN=20, confidence_threshold=0.9):
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
    params = []
    weights = []
    # extract parameter names and weights
    for s in sensitize_weight.items():
        params.append(s[0])
        weights.append(s[1])

    # sort weights descendingly
    sorted_indice = np.argsort(weights)[::-1]
    params_sorted = [params[i] for i in sorted_indice]
    weights_sorted = [weights[i] for i in sorted_indice]

    weights_cumsum = np.cumsum(np.array(weights_sorted))
    index = np.where(weights_cumsum >= confidence_threshold)[0][0]
    k = topN if topN <= index else index + 1
    confidence = weights_cumsum[k - 1]

    return (params_sorted[:k], weights_sorted[:k], confidence)


@normalFuncLog
def _loadData(data_path):
    """Load and prepare data for analyzing

    Args:
        data_path (string): path of numpy data
    Return:
        X (numpy array of shape (n,m)): training data,
        y (numpy array of shape (n,)): training target for regression,
        params (list): a list of parameter names
    """

    # extract parameter names
    knobs = np.load(data_path + '/knobs.pkl', allow_pickle=True)
    params = [k['name'] for k in knobs]

    # load data
    X = np.load(data_path + '/points.pkl', allow_pickle=True)
    X = tools.normalizePts(X, knobs)
    # load score
    bench = np.load(data_path + '/bench.pkl', allow_pickle=True)
    # find benchmark results with highest weight as the main objective for analysis
    bench_index = np.argsort([bench[k]['weight'] for k in bench.keys()])[-1]
    score = np.load(data_path + '/score.pkl', allow_pickle=True)
    y = score[:, bench_index]

    return X, y, params


@normalFuncLog
def _getStability(Z):
    """Compute stability according

    Args:
        mask (numpy array): A mask matrix with binary values for selected parameters.

    Return:
        stable: The stability of the feature selection procedure
    """
    n_trails, n_params = mask.shape
    # get empirical mean factor
    empirical_factor = float(n_trails) / (n_trails - 1)

    # get mean of mask values and compute 1st step stable score
    mask_mean = mask.mean(axis=0)
    stable = np.multiply(mask_mean, 1 - mask_mean).mean()
    # get expected value of mask as normalizing facotor
    mask_expected = mask_mean.sum()
    normalize_factor = (mask_expected / n_params) * \
        (1 - mask_expected / n_params)

    # compute stable scores
    stable = empirical_factor * stable / normalize_factor
    stable = 1 - stable
    if np.isnan(stable):
        stable = 1.0
    return stable


@normalFuncLog
def _computeStability(log, params, plot=True):
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
    n_trials = len(log.keys())
    n_params = len(params)
    # find sorted indice of params
    params_order = np.zeros(shape=(n_trials, n_params), dtype=np.float16)
    for k in range(n_trials):
        params_sorted = [params[i] for i in list(
            np.abs(log[k]['sensitivity']).argsort()[::-1])]
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
                selected_params = set([params[i] for i in topK[0]])
            else:
                selected_params = selected_params.intersection(
                    set([params[i] for i in topK[0]]))

        results['stable_I'][i] = _getStability(Z[i]) if n_trials > 1 else 1.0
        results['stable_II'][i] = float(len(selected_params)) / n

    # directories to save stable scores and parameter orders box-plot
    dump_folder_path = os.path.join(
        Config.sensi_data_dir, datetime.now().strftime("%y-%m-%d-%H-%M-%S"))

    if not os.path.exists(dump_folder_path):
        os.makedirs(dump_folder_path)

    params_name_file = os.path.join(dump_folder_path, "name.pkl")
    params_order_file = os.path.join(dump_folder_path, "order.pkl")

    params_order_median = np.median(params_order, axis=0)
    params_order_median_indice = params_order_median.argsort()
    params_order_sorted = [params[i] for i in params_order_median_indice]

    pickle.dump(params_order_sorted, open(params_name_file, 'wb+'))
    pickle.dump(params_order[:, params_order_median_indice], open(
        params_order_file, 'wb+'))


@normalFuncLog
def _sensitizeImpl(data_path, trials=0):
    """Implementation of sensitive parameter identification algorithm

    Args:
        data_path (string): path of numpy data

    Return:
        sensitize_result: a dict of parameters sensitivity scores, keys are sorted descendingly according to the scores
                          e.g., sensitize_result = {"parameter_name": float value}
    """
    X, y, params = _loadData(data_path)
    sensitize_result = _sensitizeRun(X=X, y=y, params=params, trials=trials)
    sensitize_result = _sensitizeSelect(sensitize_result)

    sleep(10)
    knobs_path = os.path.join(data_path, "knobs.pkl")
    knobs = pickle.load(open(knobs_path, 'rb'))
    return True, _parseSenstizeResult(sensitize_result, knobs)


@normalFuncLog
def _sensitizeRun(X, y, params, learner="linear", trials=0, verbose=1):
    """Implementation of sensitive parameter identification algorithm

    Args:
        X (numpy array of shape (n,m)): training data
        y (numpy array of shape (n,)): training label
        params (list of strings): a list of parameter names
        learner (string) : name of learner, only linear for now
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
        seeds = [seed]

    log = {}
    scores = np.zeros((len(seeds), len(params)))
    for i, s in enumerate(seeds):
        # initialize sensitier
        sensitizer = Analyzer(params=params, seed=s, learner=learner)
        # run sensitizer with collected data
        sensitizer.run(X, y)
        scores[i] = sensitizer.sensitivity

        if verbose > 0:
            log[i] = {}
            log[i]['seed'] = s
            log[i]['learner'] = learner
            log[i]['linear_performance'] = sensitizer.performance_linear
            log[i]['features'] = params
            log[i]['linear_sensitivity'] = sensitizer.sensitivity_set['linear']
            log[i]['sensitivity'] = sensitizer.sensitivity

    if verbose > 0:
        _computeStability(log=log, params=params)

    # compute median of sensitivity scores
    sensitivity_median = np.median(scores, axis=0)
    # sort sensitivity scores in descending order
    sensitize_result = {}
    for i in range(len(params)):
        sensitize_result[params[i]] = sensitivity_median[i]
    # return sorted(sensitize_result.items(), key=lambda d: d[1], reverse=True)
    return sensitize_result


@functionLog
def _checkFile(data_path):
    if not os.path.exists(os.path.join(data_path, "bench.pkl")):
        return False, "{} do not exits".format(os.path.join(data_path, "bench.pkl"))
    if not os.path.exists(os.path.join(data_path, "knobs.pkl")):
        return False, "{} do not exits".format(os.path.join(data_path, "knobs.pkl"))
    if not os.path.exists(os.path.join(data_path, "points.pkl")):
        return False, "{} do not exits".format(os.path.join(data_path, "points.pkl"))
    if not os.path.exists(os.path.join(data_path, "score.pkl")):
        return False, "{} do not exits".format(os.path.join(data_path, "score.pkl"))

    return True, ""


@functionLog
def _getLatestData():
    """Get latest numpy data.

    Returns:
        str: latest numpy data file path.
    """
    choice_table = []
    sub_folder_name_list = list(os.listdir(Config.tunning_data_dir))

    for _type in sub_folder_name_list:
        type_folder_path = os.path.join(Config.tunning_data_dir, _type)

        for data_folder_name in os.listdir(type_folder_path):
            data_folder_path = os.path.join(type_folder_path, data_folder_name)
            create_time = os.path.getctime(data_folder_path)
            choice_table.append((data_folder_path, create_time))

    if choice_table.__len__() == 0:
        return False, ""

    choice_table = sorted(choice_table, key=lambda x: x[1], reverse=True)
    latest_data_folder_path = choice_table[0][0]
    return True, latest_data_folder_path


@functionLog
def getDataPath(name: str):
    """Get numpy data path by data name.

    Get latest numpy data if data name is empty.

    Args:
        name (str): numpy data name.

    Returns:
        str: numpy data abs path.
    """
    if name == "":
        return _getLatestData()
    sub_folder_name_list = list(os.listdir(Config.tunning_data_dir))

    for _type in sub_folder_name_list:
        type_folder_path = os.path.join(Config.tunning_data_dir, _type)

        for data_folder_name in os.listdir(type_folder_path):
            data_name = re.split(r"[\[\]]", data_folder_name)[0]
            if data_name == name:
                data_path = os.path.join(
                    Config.tunning_data_dir, _type, data_folder_name)
                return True, data_path
    return False, ""


@functionLog
def sensitize(data_name="", trials=0):
    suc, data_path = getDataPath(data_name)
    if not suc:
        return False, "Can not find data: {}".format(data_name)

    suc, msg = _checkFile(data_path)
    if not suc:
        return False, "Check numpy data failed: {}".format(msg)

    suc, sensitize_result = _sensitizeImpl(data_path, trials)

    if not suc:
        return False, "Get sensitive parameter failed: {}".format(sensitize_result)

    return True, sensitize_result
