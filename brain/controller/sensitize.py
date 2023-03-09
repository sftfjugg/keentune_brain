import json
import pickle 

from tornado.web import RequestHandler
from multiprocessing import Process, Queue

from brain.common.config import AlgoConfig
from brain.common.pylog import logger
from brain.common.system import httpResponse
from brain.common.dataset import listData, deleteFile

from brain.algorithm.sensitize.sensitize import sensitize
from brain.controller import SENSITIZE_PROCESSES


class SensitizeProcess(Process):
    def __init__(self,
                data_name,
                trials,
                explainer,
                response_ip,
                response_port):

        super(SensitizeProcess, self).__init__()

        self.data_name = data_name
        self.trials = trials
        self.explainer = explainer
        self.response_ip = response_ip
        self.response_port = response_port

    def run(self):
        ''' process.start() '''
        logger.info("Sensitize process running, pid = {}".format(self.pid))

        try:
            sensitize_result, sensi_file = sensitize(
                data_name = self.data_name, 
                trials    = self.trials, 
                explainer = self.explainer, 
                epoch     = AlgoConfig.EPOCH, 
                topN      = AlgoConfig.TOPN,
                threshold = AlgoConfig.THRESHOLD
            )
            logger.info("[{}] Sensitize result: {}".format(self.pid, sensitize_result))

            head = ",".join([i['name'] for i in sensitize_result])
            data = pickle.load(open(sensi_file,'rb')).tolist()
            response_data = {"suc": True, "head": head, "result": data, "msg": ""}

        except Exception as e:
            logger.error("[{}] Sensitize error: {}".format(self.pid, e))
            response_data = {"suc": False, "head": "", "result": [], "msg": "{}".format(e)}

        httpResponse(response_data, self.response_ip, self.response_port, "sensitize_result")


    def terminate(self):
        ''' process.terminate() '''
        logger.info("Terminate sensitize process, pid = {}".format(self.pid))


class SensitizeHandler(RequestHandler):
    def post(self):
        global SENSITIZE_PROCESSES

        request_data = json.loads(self.request.body)
        logger.info("Get sensitize request: {}".format(request_data))
        
        try:
            p = SensitizeProcess(
                trials      = int(request_data['trials']),
                data_name   = request_data['data'],
                explainer   = request_data['explainer'],
                response_ip = request_data['resp_ip'],
                response_port = request_data['resp_port']
            )
            p.start()
            SENSITIZE_PROCESSES.append(p)
        
        except Exception as e:
            logger.error("Failed to start sensitize process: {}".format(e))
            self.write(json.dumps({
                "suc" : False, 
                "msg": "Failed to start sensitize process: {}".format(e)}))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({
                "suc" : True, 
                "msg": ""}))
            self.set_status(200)
            self.finish()


class TerminateHandler(RequestHandler):
    def get(self):
        global SENSITIZE_PROCESSES

        for p in SENSITIZE_PROCESSES:
            p.terminate()
        
        self.set_status(200)
        self.finish()


class DataDeleteHandler(RequestHandler):
    def post(self):
        try:
            data = json.loads(self.request.body)
            deleteFile(data['data'])

        except Exception as e:
            self.write(json.dumps({"suc": False, "msg": "{}".format(e)}))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({"suc": True, "msg": ""}))
            self.set_status(200)
            self.finish()


class AvaliableHandler(RequestHandler):
    def get(self):
        try:
            data_list = listData()

        except Exception as e:
            self.write(json.dumps({"suc": False, "msg": "{}".format(e)}))
            self.set_status(400)
            self.finish()
        
        else:
            self.write(json.dumps({
                "suc"   : True,
                "data"  : data_list,
                "tune"  : ["Hord", "random", "TPE", "lamcts", "bgcs"],
                "explainer" :['gp','lasso','univariate','shap','aggregated']
            }))
            self.set_status(200)
            self.finish()