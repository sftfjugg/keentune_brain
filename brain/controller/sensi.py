import json
import traceback

from brain.algorithm.sensitize.sensitize import sensitize
from brain.common.config import AlgoConfig
from target.common import pylog

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
    def _sensitizeImpl(self, data_name, trials):
        try:
            suc, sensitize_result = sensitize(
                data_name = data_name, 
                trials    = trials, 
                explainer = AlgoConfig.sensi_explainer, 
                epoch     = AlgoConfig.sensi_epoch, 
                topN      = AlgoConfig.sensi_topN,
                threshold = AlgoConfig.sensi_threshold
            )

        except Exception as e:
            return False, "{}".format(e)
        
        else:
            return suc, sensitize_result


    @coroutine
    def post(self):
        def _validField(request_data):
            assert request_data.__contains__('trials')
            assert request_data.__contains__('resp_ip')
            assert request_data.__contains__('resp_port')
            assert request_data.__contains__('data')

        request_data = json.loads(self.request.body)
        pylog.logger.info("get configure request: {}".format(request_data))

        try:
            _validField(request_data)
        
        except Exception as e:
            pylog.logger.error("Failed to response request: {}".format(e))
            self.write(json.dumps({"suc" : False, "msg": str(e)}))
            self.finish()

        else:
            self.write(json.dumps({
                "suc": True,"msg": "Sensitive parameter identification is running"}))
            self.finish()

            suc, out = yield self._sensitizeImpl(request_data['data'], int(request_data['trials']))
            if suc:
                response_data = {"suc": suc, "result": out, "msg": ""}
            else:
                response_data = {"suc": suc, "result": {}, "msg": out}

            _, _ = yield self._response(response_data, request_data['resp_ip'], request_data['resp_port'])


class sensiGraphHandler(RequestHandler):
    def get(self):
        from brain.visualization.sensiGraph import getSensiGraph
        suc, html_file_path = getSensiGraph()
        if not suc:
            self.write("get sensi graph failed:{}".format(html_file_path))
            self.set_status(200)
            self.finish()

        else:
            self.render(html_file_path)
            self.set_status(200)