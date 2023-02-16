import sys

#from pyroapi import optim
import numpy as np
from typing import List, Dict
from brain.algorithm.tunning.boBase import BOOptimizer
from keenopt.strategy.LaMCTS import LaMCTS
from keenopt.surrogate.MCTS import MCTS

class LamctsOptim(BOOptimizer):
    def __init__(self,
                 opt_name: str,
                 knobs: List,
                 baseline: Dict,
                 max_iteration: int=100,
                 sample_num: int=40,
                 batch_size: int=5,
                ):
        sample_iteration = int(np.ceil(max_iteration * 0.4))
        if sample_iteration >= batch_size:
            sample_num = sample_iteration
        else:
            sample_num = batch_size
        super().__init__(opt_name, max_iteration, knobs, baseline,sample_num)


        self.strategy = LaMCTS(
            fx_weight=np.ones(shape=(1,)),
            sample_iteration=sample_num,
            max_iteration=max_iteration,
            batch_size=batch_size,
            searchspace=self.searchspace
        )

        self.surrogate = MCTS(
            x_dim = self.searchspace.dim,
            fx_dim = 1,
        )

    def acquireImpl(self):
        return super().acquireImpl()
    
    def feedbackImpl(self, iteration: int, fx: float):
        return super().feedbackImpl(iteration, fx)

    def msg(self):
        return "Lamcts"


