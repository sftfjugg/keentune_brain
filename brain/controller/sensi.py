import json
import traceback

from brain.algorithm.sensitize.sensitize import sensitize
from brain.common.config import AlgoConfig

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
            response = http_client.fetch(HTTPRequest(
                url    = "http://{ip}:{port}/sensitize_result".format(
                            ip = response_ip, port = response_port),
                method = "POST",
                body   = json.dumps(response_data)
            ))
        except RuntimeError as e:
            return False, "{},{}".format(e, traceback.format_exc())

        except HTTPError as e:
            return False, "{},{}".format(e, traceback.format_exc())

        except Exception as e:
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
        return sensitize(
            data_name = data_name, 
            trials    = trials, 
            explainer = AlgoConfig.sensi_explainer, 
            epoch     = AlgoConfig.sensi_epoch, 
            topN      = AlgoConfig.sensi_topN,
            threshold = AlgoConfig.sensi_threshold
        )


    @coroutine
    def post(self):
        def _validField(request_data):
            assert request_data.__contains__('trials')
            assert request_data.__contains__('resp_ip')
            assert request_data.__contains__('resp_port')
            assert request_data.__contains__('data')

        request_data = json.loads(self.request.body)
        # load explainer and epoch from config
        explainer = AlgoConfig.sensi_explainer
        epoch = AlgoConfig.sensi_epoch
        topN = AlgoConfig.sensi_topN
        threshold = AlgoConfig.sensi_threshold

        try:
            _validField(request_data)
        
        except Exception as e:
            self.write(json.dumps({"suc" : False, "msg": str(e)}))
            self.finish()

        else:
            self.write(json.dumps({"suc": True,"msg": ""}))
            self.finish()

            suc, out = yield self._sensitizeImpl(request_data['data'], int(request_data['trials']))
            if suc:
                response_data = {"suc": suc, "result": out, "msg": ""}
            else:
                response_data = {"suc": suc, "result": {}, "msg": out}

            _, msg = yield self._response(response_data, request_data['resp_ip'], request_data['resp_port'])
            print(msg)


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