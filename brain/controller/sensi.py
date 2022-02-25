import json
from tornado.web import RequestHandler

from brain.common.system import HTTPPost
from brain.algorithm.sensitize.sensitize import sensitize
from brain.common.config import AlgoConfig

class sensitizeHandler(RequestHandler):
    async def post(self):
        request_data = json.loads(self.request.body)
        # load explainer and epoch from config
        explainer = AlgoConfig.sensi_explainer
        epoch = AlgoConfig.sensi_epoch
        topN = AlgoConfig.sensi_topN
        threshold = AlgoConfig.sensi_threshold

        try:
            trials    = int(request_data['trials'])
            resp_ip   = request_data['resp_ip']
            data_name = request_data['data']
            resp_port = request_data['resp_port']

        except KeyError as error_key:
            self.write(json.dumps({
                "suc": False,
                "msg": "can not find key: {}".format(error_key)

            }))
            self.set_status(400)
            self.finish()

        else:
            self.write(json.dumps({
                "suc": True,
                "msg": ""
            }))
            self.set_status(200)
            self.finish()

            try:
                suc, res = sensitize(data_name, trials)
            
            except Exception as e:
                response_data = {"suc": False, "result": {}, "msg": "{}".format(e)}
            
            else:
                if suc:
                    response_data = {
                        "suc": True, "result": res, "msg": ""}
                else:
                    response_data = {
                        "suc": False, "result": {}, "msg": res}

            await HTTPPost(
                api="sensitize_result",
                ip=resp_ip,
                port=resp_port,
                data=response_data
            )


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
