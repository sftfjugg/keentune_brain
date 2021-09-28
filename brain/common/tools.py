import os
import re
import numpy as np

from brain.common.config import Config


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
            else:
                nor_pts[_iter][index] = (
                    pts[_iter][index] - param['range'][0]) / (param['range'][1] - param['range'][0])
    return nor_pts


def dataList():
    """List tunning numpy data path.

    Returns:
        list: numpy data list. e.g.
        [
            {
                "name" : "long_throughput_0101",
                "type" : "tunning",
                "algorithm" : "ETPE"
            },
        ]
    """
    data_path_list = []
    sub_folder_name_list = list(os.listdir(Config.tunning_data_dir))
    for _type in sub_folder_name_list:
        file_list = list(os.listdir(
            os.path.join(Config.tunning_data_dir, _type)))
        for file_name in file_list:
            data_path_list.append({
                "name": re.split(r"[\[\]]", file_name)[0],
                "type": _type,
                "algorithm": re.split(r"[\[\]]", file_name)[1]
            })
    return data_path_list


def deleteFile(name):
    """Delete a numpy data by name.

    Args:
        name (str): numpy data name.
    """
    import shutil
    type_list = list(os.listdir(Config.tunning_data_dir))
    for _type in type_list:
        file_list = list(os.listdir(
            os.path.join(Config.tunning_data_dir, _type)))
        for file_name in file_list:
            _name = re.split(r"[\[\]]", file_name)[0]
            if _name == name:
                _path = os.path.join(Config.tunning_data_dir, _type, file_name)
                shutil.rmtree(_path)
