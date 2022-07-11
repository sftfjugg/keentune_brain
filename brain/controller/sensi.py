import json
import traceback
import pickle 

from brain.algorithm.sensitize.sensitize import sensitize
from brain.common.config import AlgoConfig
from brain.common import pylog

from tornado.web import RequestHandler
from tornado.httpclient import HTTPClient, HTTPRequest, HTTPError
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
from tornado.gen import coroutine


class sensitizeHandler(RequestHandler):
    executor = ThreadPoolExecutor(20)

    @run_on_executor
    def _response(self,
                  response_data:dict,
                  response_ip  :str,
                  response_port:str):
        http_client = HTTPClient()
        try:
            URL = "http://{ip}:{port}/sensitize_result".format(
                            ip = response_ip, port = response_port)
            response = http_client.fetch(HTTPRequest(
                url    = URL,
                method = "POST",
                body   = json.dumps(response_data)
            ))
            pylog.logger.info("response to {}".format(URL))
            pylog.logger.info("response data {}".format(response_data))

        except RuntimeError as e:
            pylog.logger.error("Failed to response to {}: {}".format(URL, e))
            return False, "{},{}".format(e, traceback.format_exc())

        except HTTPError as e:
            pylog.logger.error("Failed to response to {}: {}".format(URL, e))
            return False, "{},{}".format(e, traceback.format_exc())

        except Exception as e:
            pylog.logger.error("Failed to response to {}: {}".format(URL, e))
            return False, "{},{}".format(e, traceback.format_exc())

        else:
            if response.code == 200:
                return True, ""
            else:
                return False, response.reason

        finally:
            http_client.close()


    @run_on_executor
    def _sensitizeImpl(self, data_name, trials, explainer = AlgoConfig.EXPLAINER):
        try:
            suc, sensitize_result, sensi_file = sensitize(
                data_name = data_name, 
                trials    = trials, 
                explainer = explainer, 
                epoch     = AlgoConfig.EPOCH, 
                topN      = AlgoConfig.TOPN,
                threshold = AlgoConfig.THRESHOLD
            )
            pylog.logger.info("Get sensitize result: {} saved in {}.".format(sensitize_result, sensi_file))
        except Exception as e:
            return False, "{}".format(e), ""
        
        else:
            return suc, sensitize_result, sensi_file


    @coroutine
    def post(self):
        def _validField(request_data):
            assert request_data.__contains__('trials')
            assert request_data.__contains__('resp_ip')
            assert request_data.__contains__('resp_port')
            assert request_data.__contains__('data')
            assert request_data.__contains__('explainer')

        request_data = json.loads(self.request.body)
        pylog.logger.info("get sensitize request: {}".format(request_data))
        
        try:
            _validField(request_data)
        
        except Exception as e:
            pylog.logger.error("Failed to runing sensitizing algorithm: {}".format(e))
            self.write(json.dumps({
                "suc" : False, 
                "msg": str(e)}))
            self.finish()

        else:
            pylog.logger.info("Runing sensitizing algorithm.")
            self.write(json.dumps({
                "suc": True,
                "msg": "Sensitive parameter identification is running"}))
            self.finish()

            suc, sensitize_result, sensi_file_path = yield self._sensitizeImpl(
                data_name = request_data['data'], 
                trials    = int(request_data['trials']),
                explainer = request_data['explainer']
            )

            if suc:
                head = ",".join([i['name'] for i in sensitize_result])
                data = pickle.load(open(sensi_file_path,'rb')).tolist()
                response_data = {"suc": suc, "head": head, "result": data, "msg": ""}
            else:
                response_data = {"suc": suc, "head": "", "result": [], "msg": sensitize_result}

            _, _ = yield self._response(response_data, request_data['resp_ip'], request_data['resp_port'])