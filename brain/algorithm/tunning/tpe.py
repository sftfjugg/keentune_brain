import hyperopt
import numpy as np

from copy import deepcopy
from multiprocessing import Process, Pipe

from brain.common.config import AlgoConfig
from brain.algorithm.tunning.base import OptimizerUnit
from brain.common import pylog


class TPE(OptimizerUnit):
    @pylog.functionLog
    def __init__(self, 
                 opt_name: str, 
                 max_iteration: int,
                 knobs: list, 
                 baseline: dict):
                         
        """Init optimizer instance, use tpe algorithm

        Args:
            knobs (list): tuning parameters
            max_iteration (int): tuning max iteration
        """

        super(TPE, self).__init__(opt_name, max_iteration, knobs, baseline)
        
        self.trials = hyperopt.Trials()
        self.config_pipe = Pipe()
        self.loss_pipe = Pipe()
        self.process = Process(target=hyperopt.fmin, args=(
            self._observe,          # fn    : trail point -> loss
            self.__searchSpace(),   # space : search space
            hyperopt.tpe.suggest,   # algo  : `hyperopt.rand.suggest` or `hyperopt.tpe.suggest`
            self.max_iteration,     # max_evals
            None,                   # timeout: None or int
            None,                   # loss_threshold : early-stop if loss is small enough
            self.trials,            # trials : evaluation points
            np.random.RandomState(),# rstate : random seed
            False,                  # verbose
            False,                  # allow_trials_fmin
            False,                  # pass_expr_memo_ctrl
            False,                  # return_argmin: if function fmin return args min
            # points_to_evaluate: Only works if trials=None points is [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 2.0}]
            None,
            1,                      # max_queue_len: speed up parallel simulatulations
            True,                   # progress bar
            None,                   # early_stop_fn: fn to early stop
        ))
        self.process.start()


    @pylog.functionLog
    def __searchSpace(self):
        """Build hyperopt search space
        """
        search_space = {}
        for param in deepcopy(self.knobs):
            if param.__contains__('options'):
                search_space[param['name']] = hyperopt.hp.choice(
                    param['name'], param['options'])

            elif param.__contains__('sequence'):
                search_space[param['name']] = hyperopt.hp.choice(
                    param['name'], param['sequence'])

            elif param.__contains__('range') and param['dtype'] == 'int':
                step = param['step'] if param.__contains__('step') and param['step'] else 1
                while (param['range'][1] - param['range'][0]) / step >= AlgoConfig.MAX_SEARCH_SPACE:
                    step *= 2
                chioce_table = list(
                    range(param['range'][0], param['range'][1], step))
                search_space[param['name']] = hyperopt.hp.choice(
                    param['name'], chioce_table)

            elif param.__contains__('range') and param['dtype'] == 'float':
                search_space[param['name']] = hyperopt.hp.uniform(
                    param['name'], param['range'][0], param['range'][1])

            else:
                raise Exception("unsupported parameter type!")

        return search_space

    @pylog.functionLog
    def _observe(self, trail_point):
        """tuning target funciton

        Input parameters configuration called candidate, this funciton is supposed to return benchmark score.

        Args:
            trail_point (dict): A candidate parameter configuration

        Returns:
            score (float): benchmark score
        """
        for knob in deepcopy(self.knobs):
            param_name = knob['name']
            param_value = trail_point[param_name]
            if knob['dtype'] == 'int':
                trail_point[param_name] = int(param_value)
            else:
                trail_point[param_name] = param_value
        self.config_pipe[0].send(trail_point)
        loss = self.loss_pipe[1].recv()
        return loss

    @pylog.functionLog
    def acquireImpl(self):
        """Acquire a candidate from this optimizer.

        Returns:
            int  : iteration of this condidate
            dict : candidate
        """
        return self.config_pipe[1].recv(), 1.0

    @pylog.functionLog
    def feedbackImpl(self, iteration: int, loss: float):
        """Feedback a benchmark score and candidate to this optimizer

        Args:
            iteration (int) : Iteration of this condidate
            candidate (dict): candidate of this score
            score (float)   : benchmark running result
        """
        self.loss_pipe[0].send(loss)

    @pylog.functionLog
    def msg(self):
        """Get message of this optimizer.

        Returns:
            string : message of this optimizer
        """
        return "TPE"
