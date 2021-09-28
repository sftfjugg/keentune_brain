import re
import os
import time
import pickle
import numpy as np

from copy import deepcopy
from abc import ABCMeta, abstractmethod

from brain.common.config import Config
from brain.common.pylog import normalFuncLog


@normalFuncLog
def softmax(x):
    e_max = sum(np.exp(x))
    e_x = np.exp(x)
    softmax_x = e_x/e_max
    return softmax_x


@normalFuncLog
def _config2pts(config: dict, knobs: list):
    pts = []
    for param in knobs:
        _value = config[param['name']]
        if param.__contains__('options'):
            pts.append(int(param['options'].index(_value)))
        else:
            pts.append(float(_value))
    return np.array([pts])


class OptimizerUnit(metaclass=ABCMeta):
    """Tuning Optiomizer Base class

    1. Save processing data.
    2. Defination of input & output data and acquire & feedback method.
    3. Calculating loss value with weight and strict punishment.

    Attribute:
        opt_name (string) : Name and data folder of this optimizer.
        knobs ([dict]) : 
            Parameters list. e.g.
            knobs = [
                {
                    "name"  : "Continuous_parameter",
                    "range" : [0,100],
                    "step"  : 1,
                    "dtype" : "int",
                },
                {
                    "name"   : "Discrete_parameter",
                    "options": ['0','1'],
                    "dtype"  : "int",
                }
            ]
        dim (int) : Number of parameters.
        bench (dict) : 
            Configuration of benchmark.  e.g.
            bench = { 
                "Bench_1": {
                    "negative": false,  # if nagative 
                    "weight": 1.0,      # weight of the bench
                    "strict":false,     # if strict
                    "baseline":44000    # baseline of the bench
                },
            }

        H_config ([dict]) : 
            List of history config.  e.g.
            H_config = [
                {
                    "param1":value,
                    "param2":value
                }
            ]
        H_budget ([float]) : Budgets of each iteration.
        H_time (numpy.array) : 
            H_time count the brain running time, shape = (max_iteration, 4)
            column: time_acquire_start, time_acquire_return, time_feedback_start, time_feedback_end
        H_points (numpy.array) : Parameter value list for each iteration, shape = (max_iteration, dim)
        H_loss (numpy.array) : Final loss value for each iteration, shape = (max_iteration, )
        H_loss_parts (numpy.aray) : Parts loss value for each iteration, shape = (max_iteration, bench_size)

        sigma (float) : Weight of strict punishment.
        rho (float) : Increase rate of sigma.

    """

    @normalFuncLog
    def __init__(self, knobs: list, max_iteration: int, opt_name: str, opt_type: str):
        """Init optimizer instance, use tpe algorithm

        Args:
            knobs (list) : tuning parameters
            max_iteration (int) : tuning max iteration
            opt_name (string) : name and data folder of this optimizer.
        """
        self.bench = None
        self.knobs = knobs

        self.opt_type = opt_type
        self.opt_name = "{}[{}]".format(
            re.sub(r"\)", "]", re.sub(r"\(", "[", opt_name)), self.msg())
        self.iteration = -1
        self.bench_size = -1

        self.max_iteration = max_iteration
        self.dim = len(self.knobs)

        self.H_config = []
        self.H_budget = []

        self.H_time = np.zeros(shape=(self.max_iteration, 4), dtype=float)
        self.H_loss = np.zeros(shape=(self.max_iteration,), dtype=float)
        self.H_points = np.zeros(shape=(0, self.dim), dtype=float)
        self.folder_path = os.path.join(
            Config.tunning_data_dir, self.opt_type, self.opt_name)

        self.sigma = 1
        self.rho = 1.005 ** (500 / self.max_iteration)

    @normalFuncLog
    def _getLoss(self, bench_score: dict, iteration: int):
        """Calculate loss value by benchmark score

        1. Multiply with weight of bench.
        2. Invert value of bench is negative.
        3. Add punishment of "strict" is True.

        Args:
            bench_score (dict): Score dictionary of benchmark.  e.g.
                bench_score = {
                        "Bench_1": {
                            "value":45000,      # score value
                            "negative": false,  # if nagative 
                            "weight": 1.0,      # weight of the bench
                            "strict":false,     # if strict
                            "baseline":44000    # baseline of the bench
                        },
                    }
            iteration (int) : iteration determined the strict punishment weight.

        Return:
            loss_parts (list): Score list for each bench. Final loss is the sum of loss_parts.
        """
        loss_parts = []
        weight = softmax([float(bench_score[bench_name]['weight'])
                         for bench_name in self.bench.keys()])
        for i, bench_name in enumerate(self.bench.keys()):
            self.H_score[iteration][i] = float(
                bench_score[bench_name]['value'])

            if float(bench_score[bench_name]['baseline']) != 0:
                baseline = float(bench_score[bench_name]['baseline'])
            else:
                baseline = np.sum(self.H_score, axis=0)[i] / iteration

            if baseline == 0:
                _loss = 0.0
            else:
                _loss = float(bench_score[bench_name]['value']) / baseline - 1

            if not bench_score[bench_name]['negative']:
                _loss = - _loss

            if bench_score[bench_name]['strict'] and _loss > 0:
                _loss = self.sigma * _loss**2 / 2
            else:
                _loss *= weight[i] * 100
            loss_parts.append(_loss)
        self.sigma = self.sigma * self.rho
        return loss_parts

    @normalFuncLog
    def acquire(self):
        """Acquire a candidate and budget

        This method will call acquireImple() which implementd by optimizer class.
        This method saves history data and check validation of candidate.

        Returns:
            iteration (int) : This iteration returnd.
            candidate (list): 
                Candidate config of this iteration.  e.g.
                candidate = [
                    {
                        "name"  : "Continuous_parameter",
                        "range" : [0,100],
                        "step"  : 1,
                        "dtype" : "int",
                        "value" : 50,
                    },
                    {
                        "name"   : "Discrete_parameter",
                        "options": ['0','1'],
                        "dtype"  : "int",
                        "value" : '1',
                    }
                ]
            budget (float) : budget of benchmark in this iteration.
        """
        # Before acquire
        self.iteration += 1
        if self.iteration >= self.max_iteration:
            return -1, [], 0

        self.H_time[self.iteration][0] = time.time()

        # Acquire implement
        config, budget = self.acquireImpl()

        if len(config.keys()) == 0:
            return -1, [], 0

        # After acquire
        candidate = deepcopy(self.knobs)
        for param in candidate:
            param['value'] = config[param['name']]

        assert len(self.H_config) == self.iteration
        assert len(self.H_budget) == self.iteration

        self.H_budget.append(budget)
        self.H_config.append(config)

        pts = _config2pts(config, self.knobs)
        assert pts.shape[0] == 1 and pts.shape[1] == self.H_points.shape[1]
        self.H_points = np.concatenate((self.H_points, pts), axis=0)

        self.H_time[self.iteration][1] = time.time()
        return self.iteration, candidate, budget

    @abstractmethod
    def acquireImpl(self):
        """Sub-class is expected to implement this method to return config and budget

        Return:
            config (dict) : 
                Configuration dictionary of this iteration.  e.g.
                config = {
                    "param1":value,
                    "param2":value
                }

            budget (float) : Benchmark running budget.       
        """
        pass

    @normalFuncLog
    def feedback(self, iteration: int, bench_score: dict):
        """Feedback a benchmark score.

        Args:
            iteration (int) : Iteration of this condidate
            bench_score (dict) : Benchmark running result score of each bench.
        """
        self.H_time[iteration][2] = time.time()
        assert iteration < self.max_iteration
        assert iteration == self.iteration

        if self.bench is None:
            self.bench = bench_score
            self.bench_size = len(self.bench.keys())
            self.H_score = np.zeros(
                shape=(self.max_iteration, self.bench_size), dtype=float)
            self.H_loss_parts = np.zeros(
                shape=(self.max_iteration, self.bench_size), dtype=float)

        loss_parts = self._getLoss(bench_score, iteration)

        self.H_loss[iteration] = sum(loss_parts)
        self.H_loss_parts[iteration] = np.array(loss_parts)

        self.feedbackImpl(iteration, self.H_loss[iteration])
        self.H_time[iteration][3] = time.time()

        if self.iteration % 10 == 1 or self.iteration == self.max_iteration - 1:
            self.__savefile()

    @abstractmethod
    def feedbackImpl(self, iteration: int, loss: float):
        """Sub-class is expected to implement this method to receive loss value

        Args:
            iteration (int) : Iteration of this iteration
            loss (float) : Final loss of this iteration
        """
        pass

    @abstractmethod
    def msg(self):
        """Get message of this optimizer.

        Returns:
            string : message of this optimizer
        """
        pass

    @normalFuncLog
    def best(self):
        """Get bese candidate up to new

        Returns:
            candidate (dict) : candidate of this score
        """
        H_loss = self.H_loss[:len(self.H_config)]
        best_iteration = H_loss.tolist().index(min(H_loss))

        best_bench = deepcopy(self.bench)
        for i, bench_name in enumerate(self.bench.keys()):
            best_bench[bench_name]['value'] = self.H_score[best_iteration][i]

        best_config = self.H_config[best_iteration]
        best_candidate = deepcopy(self.knobs)
        for param in best_candidate:
            param['value'] = best_config[param['name']]

        return best_iteration, best_candidate, best_bench

    @normalFuncLog
    def __savefile(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        pickle.dump(self.knobs, open(os.path.join(
            self.folder_path, "knobs.pkl"), 'wb+'))
        pickle.dump(self.bench, open(os.path.join(
            self.folder_path, "bench.pkl"), 'wb+'))
        pickle.dump(self.H_time, open(os.path.join(
            self.folder_path, "time.pkl"), 'wb+'))
        pickle.dump(self.H_loss, open(os.path.join(
            self.folder_path, "loss.pkl"), 'wb+'))
        pickle.dump(self.H_score, open(os.path.join(
            self.folder_path, "score.pkl"), 'wb+'))
        pickle.dump(self.H_points, open(os.path.join(
            self.folder_path, "points.pkl"), 'wb+'))
        pickle.dump(self.H_loss_parts, open(os.path.join(
            self.folder_path, "loss_parts.pkl"), 'wb+'))
        return self.folder_path
