import numpy as np

def normalizePts(pts: np.array, knobs: list):
    """ Normalize points

    x` = (x - min) / (max - min)

    Args:
        pts (np.array): points to normalization.
        knobs (list): parameter search space.

    Returns:
        np.array: points normalized
    """
    nor_pts = np.zeros(shape=pts.shape)
    for _iter in range(pts.shape[0]):
        for index, param in enumerate(knobs):
            if param.__contains__('options'):
                nor_pts[_iter][index] = pts[_iter][index] / \
                    (len(param['options']) - 1)

            elif param.__contains__('sequence'):
                nor_pts[_iter][index] = pts[_iter][index] / \
                    (len(param['sequence']) - 1)

            elif param.__contains__('range'):
                nor_pts[_iter][index] = (
                    pts[_iter][index] - param['range'][0]) / (param['range'][1] - param['range'][0])

            else:
                raise Exception("unsupported parameter type!")
            
    return nor_pts