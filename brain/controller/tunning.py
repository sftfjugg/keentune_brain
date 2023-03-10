import json

from tornado.web import RequestHandler
from multiprocessing import Process, Queue

from brain.common.pylog import logger
from brain.controller import TUNING_PROCESS
from brain.common.config import AlgoConfig

class BrainProcess(Process):
    def __init__(self, 
                name, 
                algorithm, 
                iteration, 
                parameters, 
                baseline, 
                rule_list):
        
        super(BrainProcess, self).__init__()
        self.cmd_q, self.out_q, self.input_q = Queue(), Queue(), Queue()

        if algorithm.lower() == "tpe":
            from brain.algorithm.tunning.tpe import TPE
            self.optimizer = TPE(name, iteration, parameters, baseline)

        elif algorithm.lower() == "hord":
            from brain.algorithm.tunning.hord import HORD
            self.optimizer = HORD(name, iteration, parameters, baseline)

        elif algorithm.lower() == "random":
            from brain.algorithm.tunning.random import Random
            self.optimizer = Random(name, iteration, parameters, baseline)
        
        elif algorithm.lower() == "lamcts":
            from brain.algorithm.tunning.lamcts import LamctsOptim
            self.optimizer = LamctsOptim(name, iteration, parameters, baseline)

        elif algorithm.lower() == "bgcs":
            from brain.algorithm.tunning.bgcs import BgcsOptim
            self.optimizer = BgcsOptim(name, iteration, parameters, baseline)
        else:
            raise Exception("invalid algorithom: {}".format(algorithm))

        head_param, head_bench, head_time = self.optimizer.getDataHead()
        self.out_q.put((head_param, head_bench, head_time))

    def run(self):
        ''' process.start() '''

        logger.info("Create tuning process, pid = {}".format(self.pid))
        while True:
            cmd = self.cmd_q.get()
            if cmd == "acquire":
                iteration, candidate, budget = self.optimizer.acquire()
                logger.info("[{}] acquire candidate: {}".format(self.pid, candidate))
                self.out_q.put((iteration, candidate, budget))
            
            elif cmd == "feedback":
                iteration, bench_score = self.input_q.get(timeout = 3)
                time_data_line, benchmark_value_line = self.optimizer.feedback(
                    iteration = iteration, bench_score = bench_score)
                logger.info("[{}] feedback benchmark: {}".format(self.pid, bench_score))
                self.out_q.put((time_data_line, benchmark_value_line))

            elif cmd == "best":
                best_iteration, best_candidate, best_bench = self.optimizer.best()
                logger.info("[{}] get best candidate: {}".format(self.pid, best_candidate))
                self.out_q.put((best_iteration, best_candidate, best_bench))
    

class InitHandler(RequestHandler):
    ''' Init optimizer object '''

    def post(self):
        global TUNING_PROCESS

        request_data = json.loads(self.request.body)
        if TUNING_PROCESS is not None:
            logger.warning("Kill tuning process, pid = {}".format(TUNING_PROCESS.pid))
            TUNING_PROCESS.terminate()

        try:
            TUNING_PROCESS = BrainProcess(
                name = request_data["name"],
                algorithm = request_data["algorithm"],
                iteration = request_data["iteration"],
                parameters = request_data["parameters"],
                baseline = request_data["baseline_score"],
                rule_list = [[]]    # TODO: add rule list
            )
            # Get csv file head after initialzation.
            TUNING_PROCESS.start()
            head_param, head_bench, head_time = TUNING_PROCESS.out_q.get(timeout = 3)

        except Exception as e:
            logger.error("Initailize optimizer process failed:{}".format(e))
            self.write(json.dumps({"suc": False,
                "msg": "Initailize optimizer process failed:{}".format(e)}))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({"suc": True,"msg": "",
                    "parameters_head" : head_param,
                    "score_head"      : head_bench,
                    "time_head"       : head_time}))
            self.set_status(200)
            self.finish()


class AcquireHandler(RequestHandler):
    ''' Acquire a candidate with iteration and budget '''

    def get(self):
        global TUNING_PROCESS
        
        if TUNING_PROCESS is None:
            self.write("no tuning process running")
            self.set_status(400)
            self.finish()

        try:
            TUNING_PROCESS.cmd_q.put("acquire")
            iteration, candidate, budget = TUNING_PROCESS.out_q.get(timeout = AlgoConfig.ACQUIRE_TIMEOUT)

        except Exception as e:
            logger.error("Acquire failed: {}".format(e))
            self.write("Acquire failed:{}".format(e))
            self.set_status(400)
            self.finish()
        
        else:
            response_data = {
                "iteration" : iteration,
                "candidate" : candidate,
                "budget"    : budget,
                "parameter_value" : ",".join([str(param['value']) for param in candidate])
            }
            self.write(json.dumps(response_data))
            self.set_status(200)
            self.finish()


class FeedbackHandler(RequestHandler):
    ''' Feedback benchmark score with iteration '''

    def post(self):
        global TUNING_PROCESS
        request_data = json.loads(self.request.body)

        if TUNING_PROCESS is None:
            self.write("no tuning process running")
            self.set_status(400)
            self.finish()

        try:
            TUNING_PROCESS.input_q.put((request_data['iteration'], request_data['bench_score']))
            TUNING_PROCESS.cmd_q.put("feedback")
            time_data_line, benchmark_value_line = TUNING_PROCESS.out_q.get(timeout = AlgoConfig.FEEDBACK_TIMEOUT)

        except Exception as e:
            logger.error("Feedback failed: {}".format(e))
            self.write(json.dumps({
                "suc": False,"msg": "{}".format(e),"time_data"  : "","score_data" : ""}))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({
                "suc" : True,"msg" : "","time_data":time_data_line, "score_data":benchmark_value_line}))
            self.set_status(200)
            self.finish()


class EndHandler(RequestHandler):
    def get(self):
        global TUNING_PROCESS

        if TUNING_PROCESS is not None:
            logger.info("Terminate tunning process, pid = {}".format(TUNING_PROCESS.pid))
            TUNING_PROCESS.terminate()
        
        self.write(json.dumps({"suc": True,"msg": ""}))
        self.set_status(200)
        self.finish()


class BestHandler(RequestHandler):
    def get(self):
        global TUNING_PROCESS

        if TUNING_PROCESS is None:
            self.write("no tuning process running")
            self.set_status(400)
            self.finish()

        try:
            TUNING_PROCESS.cmd_q.put("best")
            best_iteration, best_candidate, best_bench = TUNING_PROCESS.out_q.get(timeout = AlgoConfig.ACQUIRE_TIMEOUT)

        except Exception as e:
            logger.error("Get best config failed:{}".format(e))
            self.write("Get best config failed:{}".format(e))
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