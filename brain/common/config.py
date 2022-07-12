import os
import logging

from configparser import ConfigParser

LOGLEVEL = {
    "DEBUG"     : logging.DEBUG,
    "INFO"      : logging.INFO,
    "WARNING"   : logging.WARNING,
    "ERROR"     : logging.ERROR
}

conf_file_path = "/etc/keentune/conf/brain.conf"
conf = ConfigParser()
conf.read(conf_file_path)
print("Loading config in {}".format(conf_file_path))

class AlgoConfig:
    # Auto-Tuning
    MAX_SEARCH_SPACE = int(conf['tuning']['MAX_SEARCH_SPACE'])
    SURROGATE = conf['tuning']['SURROGATE']
    STRATEGY  = conf['tuning']['STRATEGY']

    # Sensitize
    EPOCH     = int(conf['sensitize']['EPOCH'])
    TOPN      = int(conf['sensitize']['TOPN'])
    THRESHOLD = float(conf['sensitize']['THRESHOLD'])

    # TODO: remove the config item
    EXPLAINER = 'shap'


class Config:
    KEENTUNE_HOME       = conf['brain']['KEENTUNE_HOME']
    KEENTUNE_WORKSPACE  = conf['brain']['KEENTUNE_WORKSPACE']
    BRAIN_PORT          = conf['brain']['BRAIN_PORT']
    
    print("KeenTune Home: {}".format(KEENTUNE_HOME))
    print("KeenTune Workspace: {}".format(KEENTUNE_WORKSPACE))

    # workdir
    SENSI_DATA_PATH = os.path.join(KEENTUNE_WORKSPACE,'data', 'sensi_data')
    TUNE_DATA_PATH  = os.path.join(KEENTUNE_WORKSPACE,'data', 'tuning_data')

    # Log
    LOGFILE_PATH         = conf['log']['LOGFILE_PATH']
    _LOG_DIR = os.path.dirname(LOGFILE_PATH)
    CONSOLE_LEVEL        = LOGLEVEL[conf['log']['CONSOLE_LEVEL']]
    LOGFILE_LEVEL        = LOGLEVEL[conf['log']['LOGFILE_LEVEL']]
    LOGFILE_INTERVAL     = int(conf['log']['LOGFILE_INTERVAL'])
    LOGFILE_BACKUP_COUNT = int(conf['log']['LOGFILE_BACKUP_COUNT'])

    for _PATH in [KEENTUNE_WORKSPACE, TUNE_DATA_PATH, SENSI_DATA_PATH, _LOG_DIR]:
        if not os.path.exists(_PATH):
            os.makedirs(_PATH)