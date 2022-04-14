import json
from tornado.web import RequestHandler
from brain.common.pylog import APILog
from brain.common import pylog

OPTIMIZER = None

class InitHandler(RequestHandler):
    """ Init optimizer object.

    POST
        {
            "name"          : string, tuning job name
            "type"          : string, 'tuning' or 'collect'
            "algorithm"     : string, supported algorithm name, such as 'tpe', 'hord', 'deepopt'
            "iteration"     : int, tuning max iteration

            "parameters"    : list, parameters to tune
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

            "baseline_score"    : baseline benchmark values.
            {
                "benchmark_field_name" : benchmark field name, such as 'Throughput' and 'Latency'
                {
                    "base"    : list, List of baseline benchmark runing result in each time.
                    "negative"  : boolean, If negative is true, brain is supposed to reduce this benchmark field.
                    "weight"    : float, Weight if this benchmark field. 
                    "strict"    : boolean, If strict is true, worse result if this benchmark field is Unacceptable in any trade-off.
                },
            }
        }
    """
    @pylog.logit
    def __validRequest(self, request_data):
        """ check if request data is vaild

        """
        necessay_field = ['name', 'type', 'algorithm', 'iteration', 'parameters', 'baseline_score']
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


    def _createOptimizer(self, request_data):
        global OPTIMIZER
        try:
            if request_data['algorithm'] == 'tpe':
                from brain.algorithm.tunning.tpe import TPE
                OPTIMIZER = TPE(
                    opt_name = request_data['name'], 
                    opt_type = request_data['type'],
                    max_iteration = request_data['iteration'],
                    knobs = request_data['parameters'], 
                    baseline = request_data['baseline_score'])

            elif request_data['algorithm'] == 'hord':
                from brain.algorithm.tunning.hord import HORD
                OPTIMIZER = HORD(
                    opt_name = request_data['name'], 
                    opt_type = request_data['type'],
                    max_iteration = request_data['iteration'],
                    knobs = request_data['parameters'], 
                    baseline = request_data['baseline_score'])
            
            elif request_data['algorithm'] == 'random':
                from brain.algorithm.tunning.random import Random
                OPTIMIZER = Random(
                    opt_name = request_data['name'], 
                    opt_type = request_data['type'],
                    max_iteration = request_data['iteration'],
                    knobs = request_data['parameters'], 
                    baseline = request_data['baseline_score'])
            else:
                return "unkonwn algorithm {}".format(request_data['algorithm'])

        except Exception as e:
            return e

        else:
            return ""


    @APILog
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

        errormessage = self._createOptimizer(request_data)
        
        if errormessage != "" or OPTIMIZER is None:
            self.write(json.dumps({
                "suc": False,
                "msg": "init optimizer failed:{}".format(errormessage)
            }))
            self.set_status(400)
            self.finish()
            return

        else:
            # points_head, score_head, time_head
            HEAD_parameter, HEAD_benchmark, HEAD_time = OPTIMIZER.getDataHead()
            self.write(json.dumps({
                    "suc": True,
                    "msg": errormessage,
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

    @APILog
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
            
    @APILog
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
                "msg": msg
            }))
            self.set_status(400)
            self.finish()
            return
        
        try:
            OPTIMIZER.feedback(request_data['iteration'], request_data['bench_score'])

        except Exception as e:
            self.write(json.dumps({
                "suc": False,
                "msg": "{}".format(e)
            }))
            self.set_status(400)
            self.finish()
            return

        else:
            self.write(json.dumps({
                "suc": True,
                "msg": ""
            }))
            self.set_status(200)
            self.finish()
            return


class EndHandler(RequestHandler):
    @APILog
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

    @APILog
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


# class scoreGraphHandler(RequestHandler):
#     @APILog
#     def get(self):
#         suc, html_file_path = optimizer.visualScoreGraph()
#         if not suc:
#             self.write("get score graph failed:{}".format(html_file_path))
#             self.set_status(200)
#             self.finish()

#         else:
#             self.render(html_file_path)
#             self.set_status(200)


# class paramGraphHandler(RequestHandler):
#     @APILog
#     def get(self):
#         suc, html_file_path = optimizer.visualParamGraph()
#         if not suc:
#             self.write("get param graph failed:{}".format(html_file_path))
#             self.set_status(200)
#             self.finish()

#         else:
#             self.render(html_file_path)
#             self.set_status(200)
