import os
import time
import pickle
import numpy as np

from bokeh.plotting import figure, output_file, show
from brain.common.config import Config
from brain.common.pylog import functionLog

GRAPH_WIDTH = 800
GRAPH_HEIGHT = 800

@functionLog
def __getLatestSensiData():
    """ Get latest sensitivity result data
    """
    choice_table = []
    for folder_name in os.listdir(Config.sensi_data_dir):
        _path = os.path.join(Config.sensi_data_dir, folder_name)
        ctime = os.path.getctime(_path)
        choice_table.append((ctime, _path))
    choice_table = sorted(choice_table, key=lambda x: x[0])
    if len(choice_table) == 0:
        return False, ""
    else:
        return True, choice_table[-1][1]


@functionLog
def getSensiGraph():
    """Get sensitize graph.

    x axis: parameter sensitivity ranking.
    y axis: parameter name.

    Args:
        data_path (str): path of sensitivity result data.
        name_list_path (str): path of parameter name list file.

    Returns:
        str: HTML file path.
    """
    suc, data_folder_path = __getLatestSensiData()
    if not suc:
        return False, "Get latest sensitivity data failed:{}".format(data_folder_path)

    data = pickle.load(open(os.path.join(data_folder_path, "order.pkl"), 'rb'))
    name_list = pickle.load(
        open(os.path.join(data_folder_path, "name.pkl"), 'rb'))

    avg = np.mean(data, axis=0)
    sensi_sort_index = np.argsort(avg)
    name_list_sorted = [name_list[i] for i in sensi_sort_index][::-1]

    graph = figure(
        plot_width=GRAPH_WIDTH,
        plot_height=GRAPH_HEIGHT,
        title="Sensi Graph",
        x_range=(-1, len(name_list) + 2),
        y_range=name_list_sorted
    )

    graph.triangle(
        avg,
        name_list,
        size=16,
        alpha=0.8,
        color='blue'
    )

    for trail in range(data.shape[0]):
        trail_data = data[trail, ...]
        graph.circle(
            trail_data,
            name_list,
            size=8,
            alpha=0.3,
            color='red'
        )

    tmp_file_path = os.path.join(
        Config.graph_tmp_dir, "sensi_graph_{}.html".format(time.time()))
    output_file(tmp_file_path)
    show(graph)
    return True, tmp_file_path
