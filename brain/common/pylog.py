import json
import logging
import functools
import traceback

from brain.common.config import Config

from logging.handlers import TimedRotatingFileHandler


logger = logging.getLogger(__name__)

CALL_LEVEL = -1
PLACEHOLDER = " " * 4


def init(debug=False):
    logger.setLevel(Config.logfile_level)

    # Console Handler
    ch = logging.StreamHandler()
    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(Config.console_level)
    ch.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s: %(message)s"))
    logger.addHandler(ch)

    # Files Handler
    fh = TimedRotatingFileHandler(
        Config.logfile_path,
        when='D',
        interval=Config.logfile_interval,
        backupCount=Config.logfile_backup_count
    )
    fh.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s: %(message)s"))
    logger.addHandler(fh)


def functionLog(func):
    """ Auto logging decorator to function calling

    Function is excepeted return tuple as (suc, data).
    Logging arguements of this function and return value.
    Handling all exception occured in this function, logging traceback and return False.

    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        global CALL_LEVEL, PLACEHOLDER
        CALL_LEVEL += 1

        if CALL_LEVEL == 0:
            logger.info("{placeholder}[{module}.{func}] << {args} {kw}".format(
                        placeholder=PLACEHOLDER*CALL_LEVEL,
                        module=func.__module__,
                        func=func.__qualname__,
                        args=",".join(["{}".format(_arg) for _arg in args]),
                        kw=",".join(["{} = {}".format(k, v) for k, v in kw.items()])))
        else:
            logger.debug("{placeholder}[{module}.{func}] << {args} {kw}".format(
                placeholder=PLACEHOLDER*CALL_LEVEL,
                module=func.__module__,
                func=func.__qualname__,
                args=",".join(["{}".format(_arg) for _arg in args]),
                kw=",".join(["{} = {}".format(k, v) for k, v in kw.items()])))

        try:
            suc, res = func(*args, **kw)

        except Exception as e:
            logger.critical('[{module}.{func}] {trace}'.format(
                module=func.__module__,
                func=func.__qualname__,
                trace=traceback.format_exc()))
            CALL_LEVEL -= 1
            return False, e

        else:
            if suc:
                if CALL_LEVEL == 0:
                    logger.info("{placeholder}[{module}.{func}] >> {out}".format(
                        placeholder=PLACEHOLDER*CALL_LEVEL,
                        module=func.__module__,
                        func=func.__qualname__,
                        out=res))
                else:
                    logger.debug("{placeholder}[{module}.{func}] >> {out}".format(
                        placeholder=PLACEHOLDER*CALL_LEVEL,
                        module=func.__module__,
                        func=func.__qualname__,
                        out=res))

            else:
                logger.error("{placeholder}[{module}.{func}] {error}".format(
                    placeholder=PLACEHOLDER*CALL_LEVEL,
                    module=func.__module__,
                    func=func.__qualname__,
                    error=res))
            CALL_LEVEL -= 1
            return suc, res

    return wrapper


def normalFuncLog(func):
    """ More general auto logging decorator to function call

    Function can return any form of data.
    Handling all exception occured in this function, logging traceback and return None.

    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        global CALL_LEVEL, PLACEHOLDER
        CALL_LEVEL += 1

        if CALL_LEVEL == 0:
            logger.info("{placeholder}[{module}.{func}] << {args} {kw}".format(
                placeholder=PLACEHOLDER*CALL_LEVEL,
                module=func.__module__,
                func=func.__qualname__,
                args=",".join(["{}".format(_arg) for _arg in args]),
                kw=",".join(["{} = {}".format(k, v) for k, v in kw.items()])))
        else:
            logger.debug("{placeholder}[{module}.{func}] << {args} {kw}".format(
                placeholder=PLACEHOLDER*CALL_LEVEL,
                module=func.__module__,
                func=func.__qualname__,
                args=",".join(["{}".format(_arg) for _arg in args]),
                kw=",".join(["{} = {}".format(k, v) for k, v in kw.items()])))

        try:
            out = func(*args, **kw)

        except Exception as e:
            logger.critical('[{module}.{func}] {trace}'.format(
                module=func.__module__,
                func=func.__qualname__,
                trace=traceback.format_exc()))
            CALL_LEVEL -= 1
            return None

        else:
            if CALL_LEVEL == 0:
                logger.info("{placeholder}[{module}.{func}] >> {out}".format(
                    placeholder=PLACEHOLDER*CALL_LEVEL,
                    module=func.__module__,
                    func=func.__qualname__,
                    out=out))
            else:
                logger.debug("{placeholder}[{module}.{func}] >> {out}".format(
                    placeholder=PLACEHOLDER*CALL_LEVEL,
                    module=func.__module__,
                    func=func.__qualname__,
                    out=out))

            CALL_LEVEL -= 1
            return out

    return wrapper


def APILog(func):
    """ Auto logging decorator for restful api.

    Logging resquest data if api method is POST. 
    Handling all exception occured in this function, logging traceback.

    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        obj = args[0]
        if func.__name__ == "post":
            input_data = json.loads(obj.request.body)
            logger.info("{api_name} <== {input}".format(
                api_name=func.__qualname__,
                input=input_data))

        else:
            logger.info("{api_name} <== ".format(
                api_name=func.__qualname__))

        try:
            func(*args, **kw)

        except Exception as e:
            logger.critical("{api_name} {trace}".format(
                api_name=func.__qualname__,
                trace=traceback.format_exc()))

    return wrapper
