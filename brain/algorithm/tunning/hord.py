import numpy as np

from multiprocessing import Process, Pipe

from pySOT import strategy, surrogate
from pySOT.experimental_design import LatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from poap.controller import BasicWorkerThread, ThreadController

from brain.algorithm.tunning.base import OptimizerUnit
from brain.common.config import AlgoConfig
from brain.common.pylog import normalFuncLog


@normalFuncLog
def _adjustStep(param: dict):
    """ Get step of params and adjust it.

    If step is not defined in param, use default value step = 1.

    Adjust step to limit search space smaller than AlgoConfig.max_search_space

    Args:
        param (dict): parameter dictionary
    """
    if param['dtype'] in ['string', 'str']:
        return 1

    if param['dtype'] in ['float', 'double']:
        default_step = 0.1
    else:
        default_step = 1

    if param.__contains__('step'):
        step = max(default_step, param['step'])
    else:
        step = default_step

    while (param['range'][1] - param['range'][0]) / step > AlgoConfig.max_search_space:
        step *= 2

    return step


class Problem(OptimizationProblem):
    @normalFuncLog
    def __init__(self, knobs: list, max_iteration: int):
        """ Init HORD problem to solve.

        Args:
            knobs (list): parameter list.
            max_iteration (int): max tunning round.
        """
        self.dim = len(knobs)
        self.knobs = knobs
        self.max_iteration = max_iteration

        self.loss_pipe = Pipe()
        self.configuration_pipe = Pipe()

        self.buildSearchSpace(knobs)

    @normalFuncLog
    def buildSearchSpace(self, knobs: list):
        """ Build HORD parameter search space.

        Args:
            knobs (list): parameter list.
        """
        self.int_var = []
        self.cont_var = []
        self.search_space = {}

        self.lb = np.zeros(self.dim, dtype=np.int64)
        self.ub = np.zeros(self.dim, dtype=np.int64)

        for index, param in enumerate(knobs):
            if param.__contains__('range'):
                step = _adjustStep(param)
                options = [param['range'][0] + i * step for i in range(
                    1 + int((param['range'][1] - param['range'][0]) / step))]

            elif param.__contains__('options'):
                options = param['options']

            self.search_space[param['name']] = options
            self.lb[index], self.ub[index] = 0, len(options) - 1
            self.int_var.append(index)

        self.int_var = np.array(self.int_var)
        self.cont_var = np.array(self.cont_var)

    @normalFuncLog
    def eval(self, config):
        """ Evaluate a config

        Send a configuration to pipe, and waiting for benchmark running result.

        Args:
            config (dict): A parameter configuration.

        Returns:
            float: loss value of this configuration.
        """
        configuration = {}

        for index, params in enumerate(self.knobs):
            configuration[params['name']
                          ] = self.search_space[params['name']][int(config[index])]

        self.configuration_pipe[0].send(configuration)
        loss = self.loss_pipe[1].recv()
        return loss


class HORD(OptimizerUnit):
    @normalFuncLog
    def __init__(self, knobs, max_iteration, opt_name, opt_type: str):
        super(HORD, self).__init__(knobs, max_iteration, opt_name, opt_type)
        self.problem = Problem(knobs, max_iteration)

        self.strategy = self.__getStrategy()
        self.surrogate_fn = self.__getSurrogate()

        self.process = Process(target=self._fit, args=(self.problem,))
        self.process.start()

    @normalFuncLog
    def __getSurrogate(self):
        """ choose surrogate of pySOT

        Returns:
            pySOT.surrogate: surrogate in pySOT
        """
        # Choose surrogate of HORD
        if AlgoConfig.hord_surrogate == 'RBFInterpolant':
            return surrogate.RBFInterpolant

        elif AlgoConfig.hord_surrogate == 'PolyRegressor':
            return surrogate.PolyRegressor

        elif AlgoConfig.hord_surrogate == 'GPRegressor':
            return surrogate.GPRegressor

        else:
            return surrogate.RBFInterpolant

    @normalFuncLog
    def __getStrategy(self):
        """ choose strategy for HORD

        EI Strategy can only be used in combination with GPRegressor Surrogate.

        Returns:
            pySOT.strategy: strategy in pySOT
        """
        if AlgoConfig.hord_strategy == 'DYCORSStrategy':
            return strategy.DYCORSStrategy

        elif AlgoConfig.hord_strategy == 'SRBFStrategy':
            return strategy.SRBFStrategy

        elif AlgoConfig.hord_strategy == 'SOPStrategy':
            return strategy.SOPStrategy

        elif AlgoConfig.hord_strategy == 'EIStrategy' and AlgoConfig.hord_surrogate == 'GPRegressor':
            return strategy.EIStrategy

        else:
            return strategy.DYCORSStrategy

    @normalFuncLog
    def _fit(self, problem):
        self.controller = ThreadController()
        ''' create and reset surrogate '''
        self.surrogate = self.surrogate_fn(
            dim=problem.dim,
            lb=problem.lb,
            ub=problem.ub,
        )

        ''' reset surrogate'''
        self.controller.strategy = self.strategy(
            max_evals=problem.max_iteration,
            opt_prob=problem,
            exp_design=LatinHypercube(
                dim=problem.dim,
                num_pts=problem.dim * 2,
                iterations=200,
            ),
            surrogate=self.surrogate
        )

        ''' add extern-dataset '''

        worker = BasicWorkerThread(self.controller, problem.eval)
        self.controller.launch_worker(worker)

        ''' Main Loop '''
        result = self.controller.run()

        return result

    @normalFuncLog
    def acquireImpl(self):
        configuration = self.problem.configuration_pipe[1].recv()
        return configuration, 1.0

    @normalFuncLog
    def feedbackImpl(self, iteration, loss):
        if iteration <= self.dim * 2:
            self.sigma = 1
        self.problem.loss_pipe[0].send(loss)

    @normalFuncLog
    def msg(self):
        return "HORD Optimizer(v1.0)"
