import os
import pickle

from brain.common.config import Config
from brain.common.pylog import functionLog

optimizer = None


@functionLog
def init(algorithm: str, knobs: list, max_iteration: int, opt_name: str, opt_type: str):
    global optimizer

    if not optimizer is None:
        return False, "optimizer {} is running".format(optimizer.msg())

    try:
        if algorithm == "tpe":
            from brain.algorithm.tunning.tpe import TPE
            optimizer = TPE(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, opt_type=opt_type)

        elif algorithm == "hord":
            from brain.algorithm.tunning.hord import HORD
            optimizer = HORD(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, opt_type=opt_type)
        elif algorithm == 'etpe':
            from brain.algorithm.tunning.ultra import ultra
            optimizer = ultra(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, ultra_algo="ETPE", opt_type=opt_type)

        elif algorithm == 'forest':
            from brain.algorithm.tunning.ultra import ultra
            optimizer = ultra(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, ultra_algo="Forest", opt_type=opt_type)

        elif algorithm == 'gbrt':
            from brain.algorithm.tunning.ultra import ultra
            optimizer = ultra(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, ultra_algo="GBRT", opt_type=opt_type)

        elif algorithm == 'random':
            from brain.algorithm.tunning.ultra import ultra
            optimizer = ultra(knobs=knobs, max_iteration=max_iteration,
                            opt_name=opt_name, ultra_algo="Random", opt_type=opt_type)

        else:
            return False, "unknown algorithom:{}".format(algorithm)

    except ModuleNotFoundError as e:
        return False, "Failed to import dependent of algorithm {}: {}".format(algorithm, e)

    except ImportError as e:
        return False, "Failed to import dependent of algorithm {}: {}".format(algorithm, e)

    return True, optimizer.msg()


@functionLog
def __candidateCheck(candidate: list):
    for param in candidate:
        if param.__contains__('options'):
            if param['value'] not in param['options']:
                return False, "param value out of range, param:{}".format(param)
        else:
            if not (param['value'] >= param['range'][0] and param['value'] <= param['range'][1]):
                return False, "param value out of range, param:{}".format(param)
    return True, ""


@functionLog
def active():
    global optimizer
    if optimizer is None:
        return False, "No optimizer instance is activated."
    else:
        return True, optimizer.msg()


@functionLog
def acquire():
    global optimizer

    if optimizer is None:
        return False, "No optimizer instance is activated."

    iteration, candidate, budget = optimizer.acquire()
    if candidate is None:
        return True, (-1, [], 0)

    suc, msg = __candidateCheck(candidate)
    if not suc:
        return False, msg

    return True, (iteration, candidate, budget)


@functionLog
def feedback(iteration: int, score: float):
    global optimizer

    if optimizer is None:
        return False, "No optimizer instance is activated."

    optimizer.feedback(iteration, score)
    return True, ""


@functionLog
def end():
    global optimizer

    if optimizer is None:
        return False, "No optimizer instance is activated."

    del optimizer
    optimizer = None
    return True, ""


@functionLog
def getBest():
    global optimizer

    if optimizer is None:
        return False, "No optimizer instance is activated."

    best_iteration, best_candidate, best_bench = optimizer.best()
    if best_candidate is None:
        return False, "candidate is null."

    suc, msg = __candidateCheck(best_candidate)
    if not suc:
        return False, msg

    return True, (best_iteration, best_candidate, best_bench)


@functionLog
def __getLatestFile():
    """Get latest numpy data folder

    Returns:
        suc (bool) : if success to find latest data  
        path (string) : data folder path 
    """
    choice_table = []
    for tune_type in os.listdir(Config.tunning_data_dir):
        for folder_path in os.listdir(os.path.join(Config.tunning_data_dir, tune_type)):
            _path = os.path.join(Config.tunning_data_dir,
                                 tune_type, folder_path)
            ctime = os.path.getctime(_path)
            choice_table.append((ctime, _path))
    choice_table = sorted(choice_table, key=lambda x: x[0])

    if len(choice_table) == 0:
        return False, ""
    else:
        return True, choice_table[-1][1]


@functionLog
def visualScoreGraph():
    global optimizer
    from brain.visualization.tuneGraph import unionGraph

    if optimizer is None:
        suc, latest_data_path = __getLatestFile()
        if not suc:
            return False, "Find latest file failed:{}".format(latest_data_path)

        bench = pickle.load(
            open(os.path.join(latest_data_path, "bench.pkl"), 'rb'))
        score_matrix = pickle.load(
            open(os.path.join(latest_data_path, "score.pkl"), 'rb'))
        time_matrix = pickle.load(
            open(os.path.join(latest_data_path, "time.pkl"), 'rb'))
        loss_parts_matrix = pickle.load(
            open(os.path.join(latest_data_path, "loss_parts.pkl"), 'rb'))

    else:
        try:
            bench = optimizer.bench
            score_matrix = optimizer.H_score
            time_matrix = optimizer.H_time
            loss_parts_matrix = optimizer.H_loss_parts

        except AttributeError as e:
            return False, "Optimizer is running."

    html_file_path = unionGraph(
        bench=bench,
        score_matrix=score_matrix,
        time_matrix=time_matrix,
        loss_parts=loss_parts_matrix
    )

    return True, html_file_path


@functionLog
def visualParamGraph():
    global optimizer
    from brain.visualization.paramGraph import getParamGraph

    if optimizer is None:
        suc, latest_data_path = __getLatestFile()
        if not suc:
            return False, "Find latest file failed:{}".format(latest_data_path)

        knobs = pickle.load(
            open(os.path.join(latest_data_path, "knobs.pkl"), 'rb'))
        point_matrix = pickle.load(
            open(os.path.join(latest_data_path, "points.pkl"), 'rb'))
        loss_matrix = pickle.load(
            open(os.path.join(latest_data_path, "loss.pkl"), 'rb'))

    else:
        knobs = optimizer.knobs
        point_matrix = optimizer.H_points
        loss_matrix = optimizer.H_loss

    html_file_path = getParamGraph(
        points=point_matrix, knobs=knobs, loss=loss_matrix)
    return True, html_file_path
