import tornado

from brain.controller import tunning, sensi, system

from brain.common import pylog
from brain.common.config import Config

"""
AI Engine

RESTFUL API:
    /init    : Initialize the optimizer.
    /acquire : Acquire a configuration candidate.
    /feedback: Feedback benchmark performance score to latest configuration candidate.
    /best    : Get best configuration.
    /end     : End the optimizer active.

    /sensitize : Calculate sensitivity of parameters.
    /sensitize_list  : List numpy data file which can used for calculating sensitivity of parameters.
    /sensitize_delete: Remove numpy data file.

WEB API:
    /score: Show the tunning process data such as score, loss and time cast.
    /param: Show the relationship between parameters and score and the relationship between parameters and iteration.
    /sensi: Show the sensitivity rinking of parameters.

"""


def main():
    pylog.init()

    app_brain = tornado.web.Application(handlers=[
        (r"/init", tunning.InitHandler),
        (r"/acquire", tunning.AcquireHandler),
        (r"/feedback", tunning.FeedbackHandler),
        (r"/best", tunning.BestHandler),
        (r"/end", tunning.EndHandler),
        (r"/sensitize", sensi.sensitizeHandler),
        (r"/sensitize_list", system.dataListHandler),
        (r"/sensitize_delete", system.dataDeleteHandler),
    ])
    http_server_brain = tornado.httpserver.HTTPServer(app_brain)
    http_server_brain.listen(Config.brain_port)

    app_graph = tornado.web.Application(handlers=[
        (r"/", tunning.scoreGraphHandler),
        (r"/param", tunning.paramGraphHandler),
        (r"/score", tunning.scoreGraphHandler),
        (r"/sensi", sensi.sensiGraphHandler),
    ])
    http_server_graph = tornado.httpserver.HTTPServer(app_graph)
    http_server_graph.listen(Config.graph_port)

    print("KeenTune AI-Engine running...")
    tornado.ioloop.IOLoop.instance().start()
