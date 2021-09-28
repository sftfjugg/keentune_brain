from bokeh.palettes import YlGnBu
from brain.common.pylog import normalFuncLog


def __autocolor():
    return Category20[20]


@normalFuncLog
def getGradualColor(num: int):
    step = 255 / num
    gradual_color = []
    R, G, B = 255, 0, 0
    for _ in range(num):
        gradual_color.append((int(R), int(G), int(B)))
        R = max(0, R - step)
        G = min(G + step, 255)
        B = min(B + step/2, 128)
    return gradual_color


@normalFuncLog
def getGradualColor2(num: int):
    step = 255 / num
    gradual_color = []
    R, G, B = 255, 0, 0
    for _ in range(num):
        gradual_color.append((int(R), int(G), int(B)))
        R = max(0, R - step)
        G = min(G + step/2, 128)
        B = min(B + step, 255)
    return gradual_color


@normalFuncLog
def _drawLineGraph(graph, y_value, label, color, x_value=None):
    """Draw line graph with points

        value : (-1, )
        label : string
    """
    if x_value is None:
        x_value = list(range(len(y_value)))

    graph.circle(
        x_value,
        y_value,
        size=4,
        alpha=1,
        color=color,
        legend_label=label
    )

    graph.line(
        x_value,
        y_value,
        line_width=2,
        line_alpha=1,
        line_color=color,
        legend_label=label
    )


@normalFuncLog
def lineGraph(graph_name, **kwargs):
    graph = figure(plot_width=1500, plot_height=800,
                   y_range=[0, 1], title=graph_name)

    for curve_name, curve_value in kwargs.items():
        _drawLineGraph(graph, curve_value, curve_name)

    graph.legend.location = "top_left"
    graph.legend.click_policy = "hide"

    output_file(graph_name + '.html')
    show(graph)
