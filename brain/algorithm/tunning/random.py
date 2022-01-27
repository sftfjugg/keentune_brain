import numpy as np

from brain.algorithm.tunning.base import OptimizerUnit
from brain.common import pylog


class Random(OptimizerUnit):
    @pylog.logit
    def __init__(self,
                 opt_name: str,
                 opt_type: str,
                 max_iteration: int,
                 knobs: list,
                 baseline: dict):
        super(Random, self).__init__(opt_name, opt_type, max_iteration, knobs, baseline)

    @pylog.logit
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

    @pylog.logit
    def feedbackImpl(self, iteration: int, loss: float):
        pass

    @pylog.logit
    def msg(self):
        return "Random"
