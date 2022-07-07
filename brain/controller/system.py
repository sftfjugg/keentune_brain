import json

from brain.common.dataset import listData, deleteFile
from tornado.web import RequestHandler


class avaliableHandler(RequestHandler):
    def get(self):
        data_list = listData()
        resp_data = {
            "suc"   : True,
            "data"  : data_list,
            "tune"  : ['hord', 'random', 'tpe'],
            "explainer" :['gp','lasso', 'shap', 'explain', 'univariate']
        }
        self.write(json.dumps(resp_data))
        self.set_status(200)


class dataListHandler(RequestHandler):
    def get(self):
        data_list = listData()
        resp_data = {
            "suc"   : True,
            "data"  : data_list
        }
        self.write(json.dumps(resp_data))
        self.set_status(200)


class dataDeleteHandler(RequestHandler):
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
            deleteFile(data_name)
            resp_data = {
                "suc": True,
                "msg": ""
            }
            self.write(json.dumps(resp_data))
            self.set_status(200)