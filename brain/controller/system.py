import json

from tornado.web import RequestHandler

from brain.common import tools
from brain.common.pylog import APILog


class dataListHandler(RequestHandler):
    @APILog
    def get(self):
        data_list = tools.dataList()
        resp_data = {
            "suc": True,
            "data": data_list
        }
        self.write(json.dumps(resp_data))
        self.set_status(200)


class dataDeleteHandler(RequestHandler):
    @APILog
    def post(self):
        data = json.loads(self.request.body)
        try:
            data_name = data['data']

        except KeyError as error_key:
            self.write(json.dumps({
                "suc": False,
                "msg": "can not find key: {}".format(error_key)

            }))
            self.finish()
            self.set_status(400)

        else:
            tools.deleteFile(data_name)
            resp_data = {
                "suc": True,
                "msg": ""
            }
            self.write(json.dumps(resp_data))
            self.set_status(200)
