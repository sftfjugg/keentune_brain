import os
import time
import numpy as np
from collections import defaultdict

from brain.common import tools
from brain.common.config import Config
from brain.visualization.common import getGradualColor, getGradualColor2

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot

from brain.common.pylog import normalFuncLog


@normalFuncLog
def __getParamIterationGraph(points, knobs):
    """Get param iteration graph, show the relationship between parameter values and iteraion.

    x axis: normalized parameter value.
    y axis: parameter name

    Args:
        points (np.array): parameters value matrix, iteration x dim
        knobs  (dict): parameters search space, dtype and step. 

    Returns:
        bokeh.plotting.figure: param iteration graph
    """
    points = tools.normalizePts(points, knobs)
    param_name_list = [knob['name'] for knob in knobs]

    param_iteration_graph = figure(
        plot_width=1200,
        plot_height=120 * len(param_name_list),
        title="Parameter Iteration Graph",
        x_range=(-0.1, 1.1),
        y_range=param_name_list
    )

    colors = getGradualColor2(points.shape[0])

    for iteration in range(points.shape[0]):
        param_value = []
        for i in range(points.shape[1]):
            if knobs[i].__contains__('range'):
                param_value.append(points[iteration][i])
            else:
                param_value.append(
                    points[iteration][i] + (np.random.rand(1)-0.5)/10)

        param_iteration_graph.line(
            param_value,
            param_name_list,
            line_width=1,
            line_alpha=0.8,
            color=colors[iteration]
        )

    param_iteration_graph.yaxis.axis_label = "Parameters Name"
    param_iteration_graph.xaxis.axis_label = "Parameters Value"
    return param_iteration_graph


@normalFuncLog
def __getParamScoreGraph(points, knobs, loss):
    """Get param score graph, show the relationship between parameter values and score.

    x axis: normalized parameter value.
    y axis: parameter name

    Args:
        points (np.array): parameters value matrix, iteration x dim
        knobs  (dict): parameters search space, dtype and step. 
        loss   (np.array): loss value matrix, iteration x 1 

    Returns:
        bokeh.plotting.figure: param score graph
    """
    points = tools.normalizePts(points, knobs)
    param_name_list = [knob['name'] for knob in knobs]

    param_score_graph = figure(
        plot_width=1200,
        plot_height=120 * len(param_name_list),
        title="Parameter Graph",
        x_range=(-0.1, 1.1),
        y_range=param_name_list
    )
    colors = getGradualColor(points.shape[0])
    sort_index = np.argsort(loss)

    for iteration in range(points.shape[0]):
        param_value = []
        for i in range(points.shape[1]):
            if knobs[i].__contains__('range'):
                param_value.append(points[iteration][i])
            else:
                param_value.append(
                    points[iteration][i] + (np.random.rand(1)-0.5)/10)

        param_score_graph.line(
            param_value,
            param_name_list,
            line_width=1,
            line_alpha=0.8,
            color=colors[sort_index[iteration]]
        )

    param_score_graph.yaxis.axis_label = "Parameters Name"
    param_score_graph.xaxis.axis_label = "Parameters Value"
    return param_score_graph


@normalFuncLog
def getParamGraph(points, knobs, loss):
    end_iteration = points.shape[0]
    for i in range(points.shape[0]):
        if np.sum(points[i, ...]) == 0:
            end_iteration = i
            break

    param_iteration_graph = __getParamIterationGraph(
        points[:end_iteration, ...], knobs)
    param_score_graph = __getParamScoreGraph(
        points[:end_iteration, ...], knobs, loss[:end_iteration])

    tmp_file_path = os.path.join(
        Config.graph_tmp_dir, "param_graph_{}.html".format(time.time()))
    p = gridplot([[param_iteration_graph, param_score_graph]])
    output_file(tmp_file_path)
    show(p)
    return tmp_file_path
