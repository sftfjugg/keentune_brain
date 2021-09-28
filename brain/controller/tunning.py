import json

from tornado.web import RequestHandler

from brain.controller import optimizer
from brain.common.pylog import APILog


class InitHandler(RequestHandler):
    @APILog
    def post(self):
        request_data = json.loads(self.request.body)

        try:
            knobs = request_data['parameters']
            opt_name = request_data['name']
            opt_type = request_data['type']
            algorithm = request_data['algorithm']
            iteration = request_data['iteration']

        except KeyError as error_key:
            self.write(json.dumps({
                "suc": False,
                "msg": "can not find key: {}".format(error_key)

            }))
            self.finish()

        else:
            suc, msg = optimizer.init(
                algorithm=algorithm,
                knobs=knobs,
                max_iteration=iteration,
                opt_name=opt_name,
                opt_type=opt_type
            )
            if not suc:
                self.write(json.dumps({
                    "suc": False,
                    "msg": "init optimizer failed:{}".format(msg)
                }))
                self.set_status(400)
                self.finish()

            else:
                self.write(json.dumps({
                    "suc": True,
                    "msg": msg
                }))
                self.set_status(200)
                self.finish()


class AcquireHandler(RequestHandler):
    @APILog
    def get(self):
        suc, data = optimizer.acquire()
        if not suc:
            self.write("acquire config failed:{}".format(data))
            self.set_status(400)
            self.finish()

        else:
            iteration, candidate, budget = data

            response_data = {
                "iteration": iteration,
                "candidate": candidate,
                "budget": budget
            }
            self.write(json.dumps(response_data))
            self.set_status(200)
            self.finish()


class FeedbackHandler(RequestHandler):
    @APILog
    def post(self):
        request_data = json.loads(self.request.body)

        try:
            score = request_data['score']
            iteration = request_data['iteration']

        except KeyError as error_key:
            self.write(json.dumps({
                "suc": False,
                "msg": "can not find key: {}".format(error_key)
            }))
            self.finish()

        suc, msg = optimizer.feedback(iteration=iteration, score=score)

        if suc:
            self.write(json.dumps({
                "suc": True,
                "msg": ""
            }))
            self.set_status(200)
            self.finish()

        else:
            self.write(json.dumps({
                "suc": False,
                "msg": msg

            }))
            self.set_status(400)
            self.finish()


class EndHandler(RequestHandler):
    @APILog
    def get(self):
        suc, msg = optimizer.end()
        self.write(json.dumps({
            "suc": suc,
            "msg": msg
        }))
        self.set_status(200)
        self.finish()


class BestHandler(RequestHandler):
    @APILog
    def get(self):
        suc, data = optimizer.getBest()
        if not suc:
            self.write("get best config failed:{}".format(data))
            self.set_status(400)
            self.finish()
            return

        best_iteration, best_candidate, best_bench = data
        response_data = {
            "iteration": best_iteration,
            "candidate": best_candidate,
            "score": best_bench
        }
        self.write(json.dumps(response_data))
        self.set_status(200)
        self.finish()


class scoreGraphHandler(RequestHandler):
    @APILog
    def get(self):
        suc, html_file_path = optimizer.visualScoreGraph()
        if not suc:
            self.write("get score graph failed:{}".format(html_file_path))
            self.set_status(200)
            self.finish()

        else:
            self.render(html_file_path)
            self.set_status(200)


class paramGraphHandler(RequestHandler):
    @APILog
    def get(self):
        suc, html_file_path = optimizer.visualParamGraph()
        if not suc:
            self.write("get param graph failed:{}".format(html_file_path))
            self.set_status(200)
            self.finish()

        else:
            self.render(html_file_path)
            self.set_status(200)
