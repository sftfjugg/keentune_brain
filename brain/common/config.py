import os
import logging
import sys

from configparser import ConfigParser

if os.geteuid() != 0:
    print("Superuser permissions are required to run the daemon.", file=sys.stderr)
    sys.exit(1)

LOGLEVEL = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR
}

conf_file_path = "/etc/keentune/conf/brain.conf"
conf = ConfigParser()
conf.read(conf_file_path)
print("Read config: {}".format(conf_file_path))


class AlgoConfig:
    # Algorithm.base
    max_search_space = int(conf['algorithm']['max_search_space'])

    # Algorithm.hord
    hord_surrogate = conf['hord']['surrogate']
    hord_strategy = conf['hord']['strategy']

    # Sensitize
    sensi_explainer = conf['sensi']['explainer']
    sensi_trials    = int(conf['sensi']['trials'])
    sensi_epoch     = int(conf['sensi']['epoch'])
    sensi_topN    = int(conf['sensi']['topN'])
    sensi_threshold     = float(conf['sensi']['threshold'])


class Config:
    keentune_home = conf['home']['keentune_home']
    keentune_workspace = conf['home']['keentune_workspace']
    print("KeenTune Home: {}".format(keentune_home))
    print("KeenTune Workspace: {}".format(keentune_workspace))

    brain_port = conf['brain']['algo_port']
    graph_port = conf['brain']['graph_port']

    # workdir
    data_dir = os.path.join(keentune_workspace, 'data')
    graph_tmp_dir = os.path.join(data_dir, "tmp_graph")
    sensi_data_dir = os.path.join(data_dir, "sensi_data")
    tunning_data_dir = os.path.join(data_dir, "tuning_data")

    # Log
    logfile_path = conf['log']['logfile_path']
    console_level = LOGLEVEL[conf['log']['console_level']]
    logfile_level = LOGLEVEL[conf['log']['logfile_level']]
    logfile_interval = int(conf['log']['logfile_interval'])
    logfile_backup_count = int(conf['log']['logfile_backup_count'])

    if not os.path.exists(keentune_workspace):
        os.makedirs(keentune_workspace)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(graph_tmp_dir):
        os.makedirs(graph_tmp_dir)

    if not os.path.exists(tunning_data_dir):
        os.makedirs(tunning_data_dir)

    if not os.path.exists(sensi_data_dir):
        os.makedirs(sensi_data_dir)

    log_dir = os.path.dirname(logfile_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
