import sys
import numpy as np
import random
import torch
from typing import List, Dict
from brain.algorithm.tunning.boBase import BOOptimizer
from keenopt.strategy.BFSGradientDescent import BFSGradientDescent as BGD_Strategy
from keenopt.surrogate.MLP_regression import MLPRegressSurrogate
from keenopt.searchspace.searchspace import SearchSpace

class BgcsOptim(BOOptimizer):
    def __init__(self,
                 opt_name: str,
                 max_iteration: int,
                 knobs: list,
                 baseline: dict,
                 sample_num: int = 20,
                 batch_size: int = 5,
                 normalize=False,
                 ):
        sample_iteration = int(np.ceil(max_iteration * 0.4))
        if sample_iteration >= batch_size:
            sample_num = sample_iteration
        else:
            sample_num = batch_size
        super().__init__(opt_name,  max_iteration, knobs, baseline, sample_num,normalize=normalize)
        parameters = {}
        for knob in self.knobs:
            parameters[knob['name']] = knob
        self.searchspace = SearchSpace(parameters)

        self.strategy = BGD_Strategy(
            fx_weight=np.ones(shape=(1,)),
            sample_iteration=sample_num,
            searchspace=self.searchspace,
            max_iteration=max_iteration,
            batch_size=batch_size
        )

        self.surrogate = MLPRegressSurrogate(
            x_dim=self.searchspace.dim,
            fx_dim=1,
        )

    def acquireImpl(self):
        return super().acquireImpl()

    def feedbackImpl(self, iteration: int, fx: float):
        return super().feedbackImpl(iteration, fx)

    def msg(self):
        return "BGCS"





