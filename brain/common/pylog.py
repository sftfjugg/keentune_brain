import json
import logging
import functools
import traceback

from brain.common.config import Config

from logging.handlers import TimedRotatingFileHandler


logger = logging.getLogger(__name__)

logger.setLevel(Config.logfile_level)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(Config.console_level)
console_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(asctime)s: %(message)s"))
logger.addHandler(console_handler)

# Log file Handler
file_handler = TimedRotatingFileHandler(
    Config.logfile_path,
    when='D',
    interval=Config.logfile_interval,
    backupCount=Config.logfile_backup_count
)
file_handler.setLevel(Config.logfile_level)
file_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(asctime)s: %(message)s"))
logger.addHandler(file_handler)

# Error log file Handler
error_handler = TimedRotatingFileHandler(
    Config.logfile_path + '.error',
    when='D',
    interval=Config.logfile_interval,
    backupCount=Config.logfile_backup_count
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(asctime)s: %(message)s"))
logger.addHandler(error_handler)

CALL_LEVEL = -1
PLACEHOLDER = " " * 4

def logit(func):
    """ Auto logging decorator to function calling

    Function is excepeted return tuple as (suc, data).
    Logging arguements of this function and return value.
    logging traceback if exception occured in this function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        global CALL_LEVEL, PLACEHOLDER
        CALL_LEVEL += 1

        logger.debug("{placeholder}[{module}.{func}] << {args} {kw}".format(
            placeholder=PLACEHOLDER*CALL_LEVEL,
            module=func.__module__,
            func=func.__qualname__,
            args=",".join(["{}".format(_arg) for _arg in args]),
            kw=",".join(["{} = {}".format(k, v) for k, v in kw.items()])))

        try:
            out = func(*args, **kw)

        except Exception as e:
            logger.error('[{module}.{func}] {trace}'.format(
                module=func.__module__,
                func=func.__qualname__,
                trace=traceback.format_exc()))
            CALL_LEVEL -= 1
            raise e

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
