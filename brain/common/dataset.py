import os
import pickle

from brain.common.config import Config


class DataSet:
    def __init__(self, data_name):
        if data_name == "":
            self.data_path = self._getLatestData()
        else:
            self.data_path = self.__getDataPath(data_name)
        
        self.knobs_path = os.path.join(self.data_path, "knobs.pkl")
        self.knobs = pickle.load(open(self.knobs_path, "rb"))

        self.points_path = os.path.join(self.data_path, "points.pkl")
        self.points= pickle.load(open(self.points_path, "rb"))

        self.bench_path = os.path.join(self.data_path, "bench.pkl")
        self.bench = pickle.load(open(self.bench_path, "rb"))

        self.score_path = os.path.join(self.data_path, "score.pkl")
        self.score = pickle.load(open(self.score_path, "rb"))


    def _getLatestData(self):
        choice_table = []
        for data_name in os.listdir(Config.TUNE_DATA_PATH):
            data_path = os.path.join(Config.TUNE_DATA_PATH, data_name)
            create_time = os.path.getctime(data_path)
            choice_table.append((data_path, create_time))

        if choice_table.__len__() == 0:
            raise Exception("Can not find lateset data!")

        choice_table = sorted(choice_table, key=lambda x: x[1], reverse=True)
        latest_data_path = choice_table[0][0]
        return latest_data_path


    def __getDataPath(self, data_name):
        if os.path.exists(os.path.join(Config.TUNE_DATA_PATH, data_name)):
            return os.path.join(Config.TUNE_DATA_PATH, data_name)
        else:
            raise Exception("Can not find data with dataname = {}".format(data_name))


def listData():
    return [data_name for data_name in list(os.listdir(Config.TUNE_DATA_PATH))]


def deleteFile(data_name):
    if os.path.exists(os.path.join(Config.TUNE_DATA_PATH, data_name)):
        os.system("rm -rf {}".format(os.path.join(Config.TUNE_DATA_PATH, data_name)))