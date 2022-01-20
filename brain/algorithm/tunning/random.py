import numpy as np

from brain.algorithm.tunning.base import OptimizerUnit
from brain.common.pylog import normalFuncLog


class Random(OptimizerUnit):
    @normalFuncLog
    def __init__(self,
                 knobs: list,
                 max_iteration: int,
                 opt_name: str,
                 opt_type: str):
        super(Random, self).__init__(knobs, max_iteration, opt_name, opt_type)
#

    @normalFuncLog
    def acquireImpl(self):
        config = {}
        for param in self.knobs:
            if param.__contains__('range'):
                config[param['name']] = np.random.randint(
                    param['range'][0], param['range'][1])

            elif param.__contains__('options'):
                config[param['name']] = param['options'][np.random.randint(
                    0, param['options'].__len__())]

            elif param.__contains__('sequence'):
                config[param['name']] = param['sequence'][np.random.randint(
                    0, param['sequence'].__len__())]

        return config, 1.0

    @normalFuncLog
    def feedbackImpl(self, iteration: int, loss: float):
        pass

    @normalFuncLog
    def msg(self):
        return "Random"
