import json
from tornado.web import RequestHandler
from brain.common.pylog import logger

OPTIMIZER = None
#AVALIABLE_ALGORITHM = ['tpe', 'hord', 'random']
AVALIABLE_ALGORITHM = ['tpe', 'hord', 'random', 'lamcts', 'bgcs']

class InitHandler(RequestHandler):
    """ Init optimizer object.
    """
    def __validRequest(self, request_data):
        """ check if request data is vaild

        """
        necessay_field = ['name', 'algorithm', 'iteration', 'parameters', 'baseline_score']
        for _field in necessay_field:
            if not request_data.__contains__(_field):
                return False, "field '{}' is not defined!".format(_field)
        
        for param_config in request_data['parameters']:
            necessay_field_param = ['name', 'domain', 'dtype', 'base']
            for _field in necessay_field_param:
                if not param_config.__contains__(_field):
                    return False, "field '{}' is not defined in parameters".format(_field)

            if param_config.__contains__("step") and not param_config.__contains__("range"):
                return False, "invalid definition of 'step' in parameter '{}'".format(param_config['name'])
            
            if sum([param_config.__contains__("range"), param_config.__contains__("options"), param_config.__contains__("sequence")]) > 1:
                return False, "Duplicate definition of 'range', 'options' and 'sequence' in parameter '{}'".format(param_config['name'])

            if sum([param_config.__contains__("range"), param_config.__contains__("options"), param_config.__contains__("sequence")]) == 0:
                return False, "Missing definition of 'range', 'options' and 'sequence' in parameter '{}'".format(param_config['name'])
        return True, ""


    def _getOptimizer(self, request_data):
        if request_data['algorithm'].lower() not in AVALIABLE_ALGORITHM:
            raise Exception("unkonwn algorithm {}".format(request_data['algorithm']))

        if request_data['algorithm'].lower() == 'tpe':
            from brain.algorithm.tunning.tpe import TPE
            _ALGORITHM = TPE

        if request_data['algorithm'].lower() == 'hord':
            from brain.algorithm.tunning.hord import HORD
            _ALGORITHM = HORD
        
        if request_data['algorithm'].lower() == 'random':
            from brain.algorithm.tunning.random import Random
            _ALGORITHM = Random


        if request_data['algorithm'].lower() == 'lamcts':
            from brain.algorithm.tunning.lamcts import LamctsOptim
            _ALGORITHM = LamctsOptim


        if request_data['algorithm'].lower() == 'bgcs':
            from brain.algorithm.tunning.bgcs import BgcsOptim
            _ALGORITHM = BgcsOptim

        return _ALGORITHM(
            opt_name      = request_data['name'], 
            max_iteration = request_data['iteration'],
            knobs         = request_data['parameters'], 
            baseline      = request_data['baseline_score'])


    def post(self):
        global OPTIMIZER
        request_data = json.loads(self.request.body)
        vaild, message = self.__validRequest(request_data)

        if not vaild:
            self.write(json.dumps({
                "suc": False,
                "msg": "invalid request: {}".format(message)
            }))
            self.set_status(400)
            self.finish()
            return

        if OPTIMIZER is not None:
            self.write(json.dumps({
                "suc": False,
                "msg": "init optimizer failed: optimizer is runing."
            }))
            self.set_status(400)
            self.finish()
            return

        try:
            OPTIMIZER = self._getOptimizer(request_data)

        except Exception as e:
            self.write(json.dumps({
                "suc": False,
                "msg": "init optimizer failed:{}".format(e)
            }))
            self.set_status(400)
            self.finish()
            return

        else:
            HEAD_parameter, HEAD_benchmark, HEAD_time = OPTIMIZER.getDataHead()
            self.write(json.dumps({
                    "suc": True,
                    "msg": "",
                    "parameters_head" : HEAD_parameter,
                    "score_head"      : HEAD_benchmark,
                    "time_head"       : HEAD_time
                }))
            self.set_status(200)
            self.finish()
            return


class AcquireHandler(RequestHandler):
    def __vailedCandidate(self, candidate):
        for param in candidate:
            if param.__contains__('options'):
                assert param['value'] in param['options']

            elif param.__contains__('sequence'):
                assert param['value'] in param['sequence']
            
            elif param.__contains__('range'):
                assert param['value'] >= param['range'][0] and param['value'] <= param['range'][1]

    def get(self):
        global OPTIMIZER
        if OPTIMIZER is None:
            self.write("Optimizer is not active.")
            self.set_status(400)
            self.finish()
            return

        try:
            iteration, candidate, budget = OPTIMIZER.acquire()
            parameter_value_list = [str(param['value']) for param in candidate]
            parameter_value_line = ",".join(parameter_value_list)

            self.__vailedCandidate(candidate)

        except Exception as e:
            self.write("acquire config failed:{}".format(e))
            self.set_status(400)
            self.finish()
        
        else:
            response_data = {
                "iteration" : iteration,
                "candidate" : candidate,
                "budget"    : budget,
                "parameter_value" : parameter_value_line
            }
            self.write(json.dumps(response_data))
            self.set_status(200)
            self.finish()


class FeedbackHandler(RequestHandler):
    """ Feedback benchmark score of a iteration.

    POST
        {
            "iteration"     : int, iteration index
            "bench_score"   :
            {
                "Throughput": [45000,45010,49002],
                "latency99" : [99,98,100]
            }
        }
    """
    def __validRequest(self, request_data):
        for _field in ['iteration', 'bench_score']:
            if not request_data.__contains__(_field):
                return False, "field '{}' is not defined!".format(_field)
        return True, ""
            
    def post(self):
        global OPTIMIZER
        if OPTIMIZER is None:
            self.write("Optimizer is not active.")
            self.set_status(400)
            self.finish()
            return

        request_data = json.loads(self.request.body)
        valid, msg = self.__validRequest(request_data)

        if not valid:
            self.write(json.dumps({
                "suc": False,
                "msg": msg,
                "time_data"  : "",
                "score_data" : ""
            }))
            self.set_status(400)
            self.finish()
            return
        
        try:
            time_data_line, benchmark_value_line = OPTIMIZER.feedback(
                iteration = request_data['iteration'], 
                bench_score = request_data['bench_score'])

        except Exception as e:
            self.write(json.dumps({
                "suc": False,
                "msg": "{}".format(e),
                "time_data"  : "",
                "score_data" : ""
            }))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({
                "suc" : True,
                "msg" : "",
                "time_data"  : time_data_line,
                "score_data" : benchmark_value_line
            }))
            self.set_status(200)
            self.finish()


class EndHandler(RequestHandler):
    def get(self):
        global OPTIMIZER
        del OPTIMIZER
        OPTIMIZER = None
        
        self.write(json.dumps({"suc": True,"msg": ""}))
        self.set_status(200)
        self.finish()


class BestHandler(RequestHandler):
    def __vailedCandidate(self, candidate):
        for param in candidate:
            if param.__contains__('options'):
                assert param['value'] in param['options']

            elif param.__contains__('sequence'):
                assert param['value'] in param['sequence']
            
            elif param.__contains__('range'):
                assert param['value'] >= param['range'][0] and param['value'] <= param['range'][1]

    def get(self):
        global OPTIMIZER
        if OPTIMIZER is None:
            self.write("Optimizer is not active.")
            self.set_status(400)
            self.finish()
            return

        try:
            best_iteration, best_candidate, best_bench = OPTIMIZER.best()
            self.__vailedCandidate(best_candidate)

        except Exception as e:
            self.write("get best config failed:{}".format(e))
            self.set_status(400)
            self.finish()
            
        else:
            response_data = {
                "iteration"  : best_iteration,
                "candidate"  : best_candidate,
                "bench_score": best_bench
            }
            self.write(json.dumps(response_data))
            self.set_status(200)
            self.finish()
