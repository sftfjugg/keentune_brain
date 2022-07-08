import logging
import functools
import traceback
import os

from brain.common.config import Config

from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)

def _initLogger():
    logger.setLevel(Config.LOGFILE_LEVEL)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.CONSOLE_LEVEL)
    console_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s: %(message)s"))

    # Log file Handler
    file_handler = TimedRotatingFileHandler(
        filename    = Config.LOGFILE_PATH,
        when        = 'D',
        interval    = Config.LOGFILE_INTERVAL,
        backupCount = Config.LOGFILE_BACKUP_COUNT
    )
    file_handler.setLevel(Config.LOGFILE_LEVEL)
    file_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s: %(message)s"))

    # Error log file Handler
    error_handler = TimedRotatingFileHandler(
        filename    = Config.LOGFILE_PATH + '.error',
        when        = 'D',
        interval    = Config.LOGFILE_INTERVAL,
        backupCount = Config.LOGFILE_BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(asctime)s: %(message)s"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

try:
    _initLogger()
except PermissionError:
    print("[PERMISSION ERROR] NO Permissions to init Log File!")
    os._exit(0)

CALL_LEVEL = -1
PLACEHOLDER = " " * 4


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

        logger.debug("{placeholder}[{module}.{func}] << {args} {kw}".format(
                    placeholder = PLACEHOLDER*CALL_LEVEL,
                    module      = func.__module__,
                    func        = func.__qualname__,
                    args        = ",".join(["{}".format(_arg) for _arg in args]),
                    kw          = ",".join(["{} = {}".format(k, v) for k, v in kw.items()])))

        try:
            out = func(*args, **kw)

        except Exception as e:
            logger.error('[{module}.{func}] {trace}'.format(
                module  = func.__module__,
                func    = func.__qualname__,
                trace   = traceback.format_exc()))
            CALL_LEVEL -= 1

        else:
            logger.debug("{placeholder}[{module}.{func}] >> {out}".format(
                placeholder = PLACEHOLDER*CALL_LEVEL,
                module      = func.__module__,
                func        = func.__qualname__,
                out         = out
            ))

            CALL_LEVEL -= 1
            return out

    return wrapper