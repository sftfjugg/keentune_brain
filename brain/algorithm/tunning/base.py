import re
import os
import time
import pickle
import numpy as np

from copy import deepcopy
from abc import ABCMeta, abstractmethod

from brain.common.config import Config
from brain.common import pylog


@pylog.functionLog
def softmax(x):
    e_max = sum(np.exp(x))
    e_x = np.exp(x)
    softmax_x = e_x/e_max
    return softmax_x


@pylog.functionLog
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
    @pylog.functionLog
    def __init__(self, 
                opt_name: str, 
                max_iteration: int, 
                knobs: list, 
                baseline: dict):
        
        """Init optimizer instance, use tpe algorithm

        Args:
            opt_name (string)   : name and data folder of this optimizer.
            opt_type (string)   : 'tuning' or 'collect'
            max_iteration (int) : tuning max iteration
            knobs (list)        : tuning parameters.
                [
                    {
                        "name"      : string, parameter name, e.g. 'vm.overcommit_memory'
                        "domain"    : string, parameter domain, e.g. 'sysctl'
                        "range"     : list,   parameter value range, given by MIN-value and MAX-value. If field 'range' is used, definition of 'options' and 'sequence' is invalid
                        "options"   : list,   parameter value range, given by Value List. If field 'options' is used, definition of 'range' and 'sequence' is invalid
                        "sequence"  : list,   parameter value range, given by Sequential Value List. If field 'sequence' is used, definition of 'range' and 'options' is invalid
                        "step"      : int,    parameter value adjust step. Only take effect when field 'range' is used.
                        "dtype"     : string, parameter data type.
                        "base"	    : string, parameter baseline config value.
                    },
                ],
            baseline(dict)          : benchmark config and baseline value.
                {
                    "benchmark_field_name" : benchmark field name, such as 'Throughput' and 'Latency'
                    {
                        "base"    : list, List of baseline benchmark runing result in each time.
                        "negative"  : boolean, If negative is true, brain is supposed to reduce this benchmark field.
                        "weight"    : float, Weight if this benchmark field. 
                        "strict"    : boolean, If strict is true, worse result if this benchmark field is Unacceptable in any trade-off.
                    },
                }
        """
        self.bench = baseline
        self.bench_size = len(self.bench.keys())
        self.fx_weight = softmax([float(baseline[bench_name]['weight']) for bench_name in baseline.keys()])

        self.knobs = knobs
        self.dim = len(self.knobs)

        self.iteration = -1
        self.max_iteration = max_iteration

        self.H_config = []
        self.H_budget = []
        self.H_time   = np.zeros(shape=(self.max_iteration, 4), dtype=float)
        self.H_loss   = np.zeros(shape=(self.max_iteration,), dtype=float)
        self.H_points = np.zeros(shape=(0, self.dim), dtype=float)
        self.H_score  = np.zeros(shape=(self.max_iteration, self.bench_size), dtype=float)
        self.H_loss_parts = np.zeros(shape=(self.max_iteration, self.bench_size), dtype=float)

        self.opt_name = "{}[{}]".format(re.sub(r"\)", "]", re.sub(r"\(", "[", opt_name)), self.msg())
        #self.folder_path = os.path.join(Config.tunning_data_dir, self.opt_type, self.opt_name)
        self.folder_path = os.path.join(Config.TUNE_DATA_PATH, opt_name)

        self.sigma = 1
        self.rho = 1.005 ** (500 / self.max_iteration)



    @pylog.functionLog
    def _getLoss(self, bench_score: dict, iteration: int):
        """Calculate loss value by benchmark score

        1. Multiply with weight of bench.
        2. Invert value of bench is negative.
        3. Add punishment of "strict" is True.

        Args:
            bench_score (dict): Score dictionary of benchmark.  e.g.
                bench_score = {
                    "Throughput": [45000,45010,49002],
                    "latency99" : [99,98,100]
                }
            iteration (int) : iteration determined the strict punishment weight.

        Return:
            loss_parts (list): Score list for each bench. Final loss is the sum of loss_parts.
        """
        loss_parts = []
        for bench_index, bench_name in enumerate(self.bench.keys()):
            # TODO: save and compare each value rather than average.
            average_score = float(np.mean(bench_score[bench_name]))
            self.H_score[iteration][bench_index] = average_score
            baseline = float(np.mean(self.bench[bench_name]['base']))

            # Relative loss
            _loss = (average_score / baseline - 1) * self.fx_weight[bench_index] * 100

            # Reverse loss if bench is positive
            if not self.bench[bench_name]['negative']:
                _loss = -_loss
            
            # strict
            if self.bench[bench_name]['strict'] and _loss > 0:
                _loss = self.sigma * _loss**2 / 2

            loss_parts.append(_loss)

        self.sigma = self.sigma * self.rho
        return loss_parts


    @pylog.functionLog
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
        raise NotImplemented("Optimizer.acquireImpl() is not implemented.")


    @pylog.functionLog
    def feedback(self, iteration: int, bench_score: dict):
        """Feedback a benchmark score.

        Args:
            iteration (int) : Iteration of this condidate
            bench_score (dict) : Benchmark running result score of each bench.
        """
        self.H_time[iteration][2] = time.time()
        if iteration != self.iteration:
            raise Exception("iteration mismatch, iteration wanted = {}, iteration feedback = {}".format(
                            self.iteration, iteration))

        loss_parts = self._getLoss(bench_score, iteration)
        mathematical_loss = sum(loss_parts)

        self.H_loss[iteration] = sum(loss_parts)
        self.H_loss_parts[iteration] = np.array(loss_parts)
        self.feedbackImpl(iteration, self.H_loss[iteration])

        #self.feedbackImpl(iteration, list(bench_score.values())[0][0])
        self.H_time[iteration][3] = time.time()

        if self.iteration % 10 == 1 or self.iteration == self.max_iteration - 1:
            self.__savefile()

        # get *.csv data
        time_value_list = [str(timestamp) for timestamp in self.H_time[iteration]]
        time_data_line = ",".join(time_value_list)

        benchmark_value_list = [str(np.mean(score)) for score in bench_score.values()]

        benchmark_value_list = list(filter(
            lambda x: self.bench[list(self.bench.keys())[benchmark_value_list.index(x)]]['weight'] > 0,
            benchmark_value_list
        ))
        benchmark_value_list.append(str(mathematical_loss))
        benchmark_value_line = ",".join(benchmark_value_list)

        return time_data_line, benchmark_value_line


    @abstractmethod
    def feedbackImpl(self, iteration: int, loss: float):
        """Sub-class is expected to implement this method to receive loss value

        Args:
            iteration (int) : Iteration of this iteration
            loss (float) : Final loss of this iteration
        """
        raise NotImplemented("Optimizer.feedbackImpl() is not implemented.")


    @abstractmethod
    def msg(self):
        """Get message of this optimizer.

        Returns:
            string : message of this optimizer
        """
        raise NotImplemented("Optimizer.msg() is not implemented.")


    @pylog.functionLog
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

    
    @pylog.functionLog
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


    def assess(self, config):
        """Assess the potential performance of config, if it's against experts' knowledge, retrieve this config

        Args:
            config = {
                    "param1":value,
                    "param2":
            }
        """
        raise NotImplementedError
    

    def early_stop(self, N=10, threshold=0.01):
        """Stop tuning optimization if the score did not improve in successive `N` steps
        
        """
        if self.iteration < N:
            return False
        best_loss = min(self.H_loss[:self.iteration+1])
        for i in range(self.iteration - N + 1, self.iteration+1):
            if self.H_loss[i] == best_loss:
                return False
        return True

    def getDataHead(self):
        """ Get head of parameter_value.csv, score.csv and time.csv

            Return the head of *.csv data after instance already initialized, *.csv files is supposed to be saved by keentuned.
        """
        parameter_name_list = [param['name'] for param in self.knobs]
        HEAD_parameter = ",".join(parameter_name_list)

        benchmark_name_list = [bench_name for bench_name in self.bench.keys() if self.bench[bench_name]['weight'] > 0]
        benchmark_name_list.append("mathematical_loss")
        HEAD_benchmark = ",".join(benchmark_name_list)

        HEAD_time = "acquire_start,acquire_end,feedback_start,feedback_end"

        return HEAD_parameter, HEAD_benchmark, HEAD_time
