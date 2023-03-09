import tornado
import os

from brain.controller import tunning, sensitize
from brain.common.config import Config

"""
AI Engine

RESTFUL API:
    /init    : Initialize the optimizer.
    /acquire : Acquire a configuration candidate.
    /feedback: Feedback benchmark performance score to latest configuration candidate.
    /best    : Get best configuration.
    /end     : End the optimizer active.
WEB API
    /sensitize : Calculate sensitivity of parameters.
    /sensitize_list  : List numpy data file which can used for calculating sensitivity of parameters.
    /sensitize_delete: Remove numpy data file.
"""


def main():
    app_brain = tornado.web.Application(handlers=[
        (r"/init", tunning.InitHandler),
        (r"/acquire", tunning.AcquireHandler),
        (r"/feedback", tunning.FeedbackHandler),
        (r"/best", tunning.BestHandler),
        (r"/end", tunning.EndHandler),
        (r"/sensitize", sensitize.SensitizeHandler),
        (r"/avaliable", sensitize.AvaliableHandler),
        (r"/terminate", sensitize.TerminateHandler),
        # (r"/sensitize_list", sensitize.dataListHandler),
        (r"/sensitize_delete", sensitize.DataDeleteHandler),
    ])
    http_server_brain = tornado.httpserver.HTTPServer(app_brain)
    http_server_brain.listen(Config.BRAIN_PORT)
    
    print("KeenTune AI-Engine running...")
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(0)