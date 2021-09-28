from ultraopt.hdl import hdl2cs
from ultraopt.optimizer import ETPEOptimizer, RandomOptimizer
from ultraopt.optimizer import ForestOptimizer, GBRTOptimizer

from brain.algorithm.tunning.base import OptimizerUnit

from brain.common.pylog import normalFuncLog


@normalFuncLog
def ultraSearchSpace(knobs: list):
    """Get parameters search space defined in ultraopt.

    Args:
        knobs ([dict]) : parameters list. e.g.
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

    Returns:
        HDL(dict): parameters search space defined in ultraopt. e.g.
            HDL = {
                "hp_choice": {"_type": "choice", "_value": ["apple", "pear", "grape"]},
                "hp_ordinal": {"_type": "ordinal", "_value": ["Primary school", "Middle school", "University"] },
                "hp_uniform": {"_type": "uniform", "_value": [0, 100]},
                "hp_quniform": {"_type": "quniform", "_value": [0, 100, 20]},
                "hp_int_uniform": {"_type": "int_uniform", "_value": [0, 10]},
                "hp_int_quniform": {"_type": "int_quniform", "_value": [0, 10, 2]},
            }

    See: https://auto-flow.github.io/ultraopt/zh/_tutorials/02._Multiple_Parameters.html

            |---------------------------------------------|
            |     _type      |     _value     |  example  |
            |---------------------------------------------|
            | "choice"       |    options     | ['0','1'] |
            | "ordinal"      |    sequence    | ['a','b'] |
            | "uniform"      | [low, high]    | [0,100]   |
            | "quniform"     | [low, high, q] | [0,100,1] |
            | "int_uniform"  | [low, high]    | [0,10]    |
            | "int_quniform" | [low, high, q] | [0,10]    |
            |---------------------------------------------|

    """
    HDL = {}
    for knob in knobs:
        if knob.__contains__('range'):
            if knob['dtype'] in ['int', 'long']:
                knob['range'][0] = int(knob['range'][0])
                knob['range'][1] = min(2 ** 62, int(knob['range'][1]))

            elif knob['dtype'] == 'float':
                knob['range'][0] = float(knob['range'][0])
                knob['range'][1] = min(2 ** 62, float(knob['range'][1]))

            if knob.__contains__('dtype') and knob['dtype'] in ['float', 'double']:
                _type = "uniform"
                _value = [knob['range'][0], knob['range'][1]]

            elif knob.__contains__('step'):
                if knob['step'] == knob['range'][1] - knob['range'][0]:
                    _type = "ordinal"
                    _value = [knob['range'][0], knob['range'][1]]

                else:
                    _type = "int_quniform"
                    _step = knob['step']
                    _lb = knob['range'][0]
                    _ub = int((knob['range'][1] - knob['range']
                              [0])/_step)*_step + knob['range'][0]
                    _value = [_lb, _ub, _step]

            else:
                _type = "int_uniform"
                _value = [knob['range'][0], knob['range'][1]]

        else:
            if all([option.isdigit() for option in knob['options']]) and len(knob['options']) > 2:
                _type = "ordinal"
                _value = knob['options']
            else:
                _type = "choice"
                _value = knob['options']

        _hdl = {
            "_type": _type,
            "_value": _value
        }
        HDL[knob['name']] = _hdl
    return HDL


class ultra(OptimizerUnit):
    """Optimizer object implement in ultraopt.

    ultraopt Docs: https://auto-flow.github.io/ultraopt/zh/index.html

    ultraopt provide 4 parameter optimizing algorithm: ETPE, Forest, GBRT and Random.

    ETPE   : Embedding-Tree-Parzen-Estimator, created by author of ultraopt.
    Forest : Bayes Optimize algorithm, from scikit-optimize import skopt.learning.forest.
    GBRT   : Bayes Optimize algorithm, base on Gradient Boosting Resgression Tree, from from scikit-optimize import skopt.learning.gbrt.

    Attribute:
        ultra_algo : algorithm name, "ETPE", "Random", "Forest" or "GBRT", default to "ETPE"
        optimizer: optimizer object. 

    """
    @normalFuncLog
    def __init__(self, knobs: list, max_iteration: int, opt_name: str, opt_type: str, ultra_algo="ETPE"):
        """Initialize Optimizer object implement in ultraopt.

        Instantiating optimizer object and parameter search space.

        Args:
            knobs ([dict]) : parameters list.
            max_iteration (int)  : max iteration number
            opt_name (str) : name of this optimizer
            ultra_algo (str, optional): algorithm name. Defaults to "ETPE".
        """
        self.ultra_algo = ultra_algo
        super(ultra, self).__init__(knobs, max_iteration, opt_name, opt_type)

        if ultra_algo == "ETPE":
            self.optimizer = ETPEOptimizer()

        elif ultra_algo == "Random":
            self.optimizer = RandomOptimizer()

        elif ultra_algo == "Forest":
            self.optimizer = ForestOptimizer()

        elif ultra_algo == "GBRT":
            self.optimizer = GBRTOptimizer()

        else:
            self.optimizer = ETPEOptimizer()

        self.optimizer.initialize(hdl2cs(ultraSearchSpace(knobs)))

    @normalFuncLog
    def acquireImpl(self):
        """Get candidate to run benchmark.

        Save config to H_config.

        Returns:
            iteration (int)  : iteration of this candidate returned.
            candidate (dict) : candidate config.
            budget (float)   : budget of this candidate to run benchmark.            
        """
        config, _ = self.optimizer.ask()
        return config, 1.0

    @normalFuncLog
    def feedbackImpl(self, iteration: int, loss: float):
        """Feedback score and candidate loss.

        Rewrite candidate from H_config, becaues ultra do not accept changed config.
        TODO: Check if config is changed.

        Args:
            iteration (int)  : iteration if this candidate.
            loss (float)    : benchmark loss of this iteration.
        """
        config = self.H_config[iteration]
        self.optimizer.tell(config, loss)

    @normalFuncLog
    def msg(self):
        return self.ultra_algo