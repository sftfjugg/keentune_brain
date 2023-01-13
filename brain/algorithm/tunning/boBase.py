import os  
import pickle
import numpy as np  
import time

from brain.algorithm.tunning.base import OptimizerUnit
from brain.common import pylog
from keenopt.sample_func.sample_func import to_unit_cube
from keenopt.sample_func.sample_func import random as randfunc
from keenopt.sample_func.sample_func import from_unit_cube
from keenopt.searchspace.searchspace import SearchSpace

from brain.common import pylog

class BOOptimizer(OptimizerUnit):
    @pylog.functionLog
    def __init__(self, 
                 opt_name: str,
                 max_iteration: int,
                 knobs: list,
                 baseline: dict,
                 sample_num: int=40,
                 initialize_epoch: int=50,
                 update_epoch: int=10,
                 train_interval: int=5,
                 train_decay: int=100,
                 normalize=True,
                 sample_func=randfunc,
                 strategy=None,
                 surrogate=None,):
        
        """ initialize optmizer object.

        Args:
            searchspace (keenopt.searchspace):  Parameter searchspace.
            surrogate (keenopt.surrogate):      Surrogate model.
            strategy (keenopt.strategy):        Coordinate search strategy.
            sample_func (keenopt.sample_func):  Pre-sampling function.
            sample_num (int):                   Pre-sampling iteration.
            max_iteration (int):                Tunning iteration.
            initialize_epoch(int, optional):    Epoch to initialize neural network model.
            update_epoch(int, optional):        Epoch to update neural network model. 
        """
        
        super().__init__(opt_name,  max_iteration, knobs, baseline)
        self.init_search_space()
        
        self.strategy    = strategy
        self.surrogate   = surrogate
        
        self.sample_num    = sample_num
        self.sample_func   = sample_func
        self.update_epoch  = update_epoch
        self.train_interval = train_interval
        self.train_decay = train_decay
        self.normalize = normalize
        self.initialize_epoch = initialize_epoch
        
        self.pts_queue = None

        # assert self.pts_queue.shape[1] == self.searchspace.dim
        
        self.H_x = np.zeros(shape=(0, self.searchspace.dim))
        self.H_fx = np.zeros(shape = (0, 1))

        self.H_pred_fx = np.zeros(shape =(0, 1))
        
        self.untrain_x = np.zeros(shape = (0, self.searchspace.dim))
        self.untrain_fx = np.zeros(shape = (0, 1))
        self.model_initialized = False


    def init_search_space(self):
        """Initialize search space
        """
        parameters = {}
        for knob in self.knobs:
            parameters[knob['name']] = knob
        self.searchspace = SearchSpace(parameters)

    @pylog.functionLog
    def acquireConfiguration(self):
        """ Get a configuration 

        If queue is empty:
            1. Training surrogate model.
            2. Generating configuration points by coordinate search strategy and save to queue 
        
        Else:
            pop a configuration points from queue and return.
        
        Returns:
            np.array: configuration point.
        """
        if self.iteration > self.max_iteration:
            return None
        
        if self.pts_queue is None:
            self.pts_queue = self.sample_func(
                pts_num = self.sample_num,
                searchspace = self.searchspace
            )
            if self.normalize:
                self.pts_queue = to_unit_cube(self.searchspace, self.pts_queue)

        if self.iteration > self.sample_num and self.untrain_x.shape[0] >= self.train_interval:
            if not self.model_initialized:
                tic = time.time()
                self.surrogate._fit(
                    x_train = self.H_x,
                    fx_train = self.H_fx,
                    epoch = self.initialize_epoch,
                    decay=self.train_decay,
                    reset_model = True,
                    pre_train = True)
                self.model_initialized = True
            else:
                tic = time.time()
                self.surrogate._fit(
                    x_train = self.H_x,
                    fx_train = self.H_fx,
                    epoch = self.update_epoch,
                    decay = self.train_decay,
                    reset_model = False,
                    pre_train = False)
            # self.surrogate._fit(
            #     x_train = self.H_x,
            #     fx_train = self.H_fx,
            #     epoch = self.initialize_epoch,
            #     reset_model = True,
            #     pre_train = True)

            self.untrain_x = np.zeros(shape = (0, self.surrogate.x_dim))
            self.untrain_fx = np.zeros(shape = (0, self.surrogate.fx_dim))
            
        if self.pts_queue.shape[0] == 0:
            batch_points = np.zeros(shape = (0, self.H_x.shape[1]))
            
            while batch_points.shape[0] == 0:
                batch_points = self.strategy.search(
                    x = self.H_x,
                    fx = self.H_fx,
                    iteration = self.iteration,
                    surrogate = self.surrogate)
                if batch_points.shape[0] == 0:
                    tic = time.time()
                    self.surrogate._fit(
                        x_train = self.H_x,
                        fx_train = self.H_fx,
                        epoch = self.initialize_epoch,
                        reset_model = True,
                        pre_train = True)
            
            self.pts_queue = np.concatenate((self.pts_queue, batch_points), axis=0)
            #if self.normalize:
            #    self.pts_queue = to_unit_cube(self.searchspace, self.pts_queue)
        # return self.pts_queue[0,...]
        x = self.pts_queue[0]
        return x

    
    def acquireImpl(self):
        x = self.acquireConfiguration()
        if self.normalize:
            x = from_unit_cube(self.searchspace, x)
        config = self.searchspace.pts2Configuration(x)
        return config, 1.0

        
        
    @pylog.functionLog
    def feedbackImpl(self, iteration: int, fx: float):
        """ Feedback benchmark score

        Args:
            score (float or np.array): benchmark result
        """
        if isinstance(fx, float) or isinstance(fx, int):
            fx = np.array([fx])
        if fx.shape.__len__() == 1:
            fx = np.expand_dims(fx,axis=0)
            
        # self.iteration += 1
        
        x = self.pts_queue[0:1,...]
        self.pts_queue = self.pts_queue[1:,...]
        
        self.H_x = np.concatenate((self.H_x, x), axis=0)
        self.H_fx = np.concatenate((self.H_fx, fx), axis=0)

        self.H_pred_fx  = np.concatenate(
            (self.H_pred_fx, self.surrogate.eval(x)), axis=0)
        self.untrain_x  = np.concatenate((self.untrain_x, x), axis=0)
        self.untrain_fx = np.concatenate((self.untrain_fx, fx), axis=0)

    
    @pylog.functionLog
    def dump(self, dump_path: str):
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        
        fx_path = os.path.join(dump_path,"fx.pkl")
        pickle.dump(self.H_fx, open(fx_path,'wb'))

        pred_fx_path = os.path.join(dump_path,"pred_fx.pkl")
        pickle.dump(self.H_pred_fx, open(pred_fx_path,'wb'))
        
        # surrogate_path = os.path.join(dump_path,"surrogate.pkl")
        # pickle.dump(self.surrogate, open(surrogate_path,'wb'))
