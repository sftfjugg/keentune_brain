import os
import time
import numpy as np

from brain.common.config import Config
from brain.visualization.common import drawLineGraph
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from brain.common.pylog import normalFuncLog

COLORS = [
    "#1E90FF",
    "#00EE76",
    "#FFD700",
    "#FF6347",
    "#363636",
    "#FF34B3"
    "#8470FF",
]

GRAPH_WIDTH = 1200
GRAPH_HEIGHT = 800


@normalFuncLog
def getScoreTimeGraph(score_matrix: np.array, time_matrix: np.array, bench: dict):
    """ Get score-time graph.

    x axis: Time stamp from start of this tunning.
    y axis: The best score at every moment.

    Args:
        score_matrix (np.array): score of each benchmark at each iteration.
        time_matrix  (np.array): time cast matrix at each iteration.
        bench (dict): bench config dict.

    Returns:
        bokeh.plotting.figure: score-time graph
    """

    iteration_end = min(score_matrix.shape[0], time_matrix.shape[0])

    # parse time matrix to x axis value
    x_value = [np.max(time_matrix[:i+1, 3]) - time_matrix[0][0]
               for i in range(iteration_end)]

    # draw score-time graph
    score_time_graph = figure(
        plot_width=GRAPH_WIDTH,
        plot_height=GRAPH_HEIGHT,
        title="BestScore-Time Graph")

    label = list(bench.keys())
    for i in range(score_matrix.shape[1]):
        baseline = bench[label[i]]['baseline']
        if bench[label[i]]['negative']:
            best_score = [np.min(score_matrix[:j+1, i]) /
                          baseline for j in range(score_matrix.shape[0])]
        else:
            best_score = [np.max(score_matrix[:j+1, i]) /
                          baseline for j in range(score_matrix.shape[0])]

        drawLineGraph(score_time_graph, best_score,
                      label[i], COLORS[i % len(COLORS)], x_value=x_value)

    score_time_graph.legend.location = "top_left"
    score_time_graph.legend.click_policy = "hide"
    score_time_graph.yaxis.axis_label = "Time Cost"
    score_time_graph.xaxis.axis_label = "Best Score"

    return score_time_graph


@normalFuncLog
def getScoreIterationGraph(score_matrix: np.array, bench: dict):
    """Get Score-iteration graph.

    Args:
        score_matrix (np.array): score of each benchmark at each moment.
        bench (dict): bench config dict.

    Returns:
        bokeh.plotting.figure: score-iteration graph
    """
    score_graph = figure(
        plot_width=GRAPH_WIDTH,
        plot_height=GRAPH_HEIGHT,
        title="Score-iteration Graph")

    label = list(bench.keys())
    for i in range(len(label)):
        baseline = bench[label[i]]['baseline']
        drawLineGraph(
            score_graph, score_matrix[..., i]/baseline, label[i], COLORS[i % len(COLORS)])

    score_graph.legend.location = "top_left"
    score_graph.legend.click_policy = "hide"
    score_graph.yaxis.axis_label = "Iteration"
    score_graph.xaxis.axis_label = "Benchmark Score"
    return score_graph


@normalFuncLog
def getTimeIterationGraph(time_matrix: np.array):
    """ Time cast graph.

    Args:
        time_matrix  (np.array): time cast matrix at each iteration.

    Returns:
        bokeh.plotting.figure: time cast-iteration graph
    """
    time_graph = figure(
        plot_width=GRAPH_WIDTH,
        plot_height=GRAPH_HEIGHT,
        title="Time Graph")

    acquire_time = time_matrix[..., 1] - time_matrix[..., 0]
    drawLineGraph(time_graph, acquire_time, "acquire_time", COLORS[0])

    benchmark_time = time_matrix[..., 2] - time_matrix[..., 1]
    drawLineGraph(time_graph, benchmark_time, "benchmark_time", COLORS[1])

    feedback_time = time_matrix[..., 3] - time_matrix[..., 2]
    drawLineGraph(time_graph, feedback_time, "feedback_time", COLORS[2])

    time_graph.legend.location = "top_left"
    time_graph.legend.click_policy = "hide"
    time_graph.yaxis.axis_label = "Iteration"
    time_graph.xaxis.axis_label = "Time Cost"
    return time_graph


@normalFuncLog
def getLossIterationGraph(bench: dict, loss_parts: np.array):
    """ Loss-iteration graph

    Args:
        bench (dict): bench config dict.
        loss_parts (np.array): loss of each benchmark at each iteration.

    Returns:
        bokeh.plotting.figure: time loss-iteration graph
    """
    loss_graph = figure(
        plot_width=GRAPH_WIDTH,
        plot_height=GRAPH_HEIGHT,
        title="Loss Graph")

    label = list(bench.keys())
    for i in range(loss_parts.shape[1]):
        drawLineGraph(
            loss_graph, loss_parts[..., i], label[i], COLORS[i % len(COLORS)])

    loss_graph.legend.location = "top_left"
    loss_graph.legend.click_policy = "hide"
    loss_graph.yaxis.axis_label = "Iteration"
    loss_graph.xaxis.axis_label = "Loss Value"
    return loss_graph


@normalFuncLog
def unionGraph(bench: dict, score_matrix: np.array, time_matrix: np.array, loss_parts: np.array):
    """ Get union graph contains score-iteration graph, time-iteration graph, loss-iteration graph and best score-time graph.

    Args:
        bench (dict): bench config dict.
        score_matrix (np.array): score of each benchmark at each moment.
        time_matrix  (np.array): time cast matrix at each iteration.
        loss_parts   (np.array): loss of each benchmark at each iteration.

    Returns:
        bokeh.plotting.figure: union graph contains score-iteration graph, time-iteration graph, loss-iteration graph and best score-time graph.
    """
    end_iteration = score_matrix.shape[0]
    for i in range(score_matrix.shape[0]):
        if np.sum(score_matrix[i, ...]) == 0:
            end_iteration = i
            break

    tmp_file_path = os.path.join(
        Config.graph_tmp_dir, "union_graph_{}.html".format(time.time()))

    s1 = getScoreIterationGraph(score_matrix[:end_iteration, ...], bench)
    s2 = getLossIterationGraph(bench, loss_parts[:end_iteration, ...])
    s3 = getTimeIterationGraph(time_matrix[:end_iteration, ...])
    s4 = getScoreTimeGraph(
        score_matrix[:end_iteration, ...], time_matrix[:end_iteration, ...], bench)

    p = gridplot([[s1, s2], [s3, s4]])
    output_file(tmp_file_path)
    show(p)
    return tmp_file_path
