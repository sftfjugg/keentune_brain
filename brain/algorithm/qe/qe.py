import os
import argparse
import random
import numpy as np
from copy import copy
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

KEENTUNE_WORKSPACE  = "./"
QE_DATA_PATH = os.path.join(KEENTUNE_WORKSPACE,'data', 'qe_data')


def _loadData(data_path,base):
    """Load and prepare data for analyzing

        Args:
            data_path (string): path of numpy data
        Return:
            X (numpy array of shape (n,m)): training data,
            y (numpy array of shape (n,)): training target for regression,
            params (list): a list of parameter names
        """

    default_params = base["param"]
    x_base = np.array(base["x"]).reshape(1,-1)
    y_base = base["y"]

    # extract parameter names
    knobs = np.load(data_path + '/knobs.pkl', allow_pickle=True)
    params_raw = [k['name'] for k in knobs]

    #p_idx = [default_params.index(p.partition("@")[0]) for p in params]
    p_idx = []
    for p1 in default_params:
        for j, p2 in enumerate(params_raw):
            if p1 == p2.partition("@")[0]:
                p_idx.append(j)

    params = [params_raw[i] for i in p_idx]
    # load data
    X = np.load(data_path + '/points.pkl', allow_pickle=True)
    X = X[:, p_idx]
    X = np.vstack([x_base, X])

    # load score
    bench = np.load(data_path + '/bench.pkl', allow_pickle=True)
    # find benchmark results with highest weight as the main objective for analysis
    bench_index = np.argsort([bench[k]['weight'] for k in bench.keys()])[-1]
    score = np.load(data_path + '/score.pkl', allow_pickle=True)
    y = score[:, bench_index]
    y = np.hstack([y_base, y])

    return X, y, params


def _segment(X, y, param, figname, n_seg=3):
    from sklearn.metrics import mean_absolute_percentage_error as mape
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    from sklearn.tree import DecisionTreeRegressor

    # n_seg: segmented linear regression parameters
    sort_idx = np.argsort(X)
    X = X[sort_idx].reshape(-1, 1)
    y = y[sort_idx].reshape(-1, 1)

    kernel = kernels.Matern()
    # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
    gpr.fit(X, y)
    y_pred, std_pred = gpr.predict(X, return_std=True)
    mape_error = mape(y, y_pred)

    X = X.squeeze()
    y_pred = y_pred.squeeze()

    xs = X + np.random.random(X.shape[0]) * 1e-4
    dys = np.gradient(y_pred, xs)
    idx = np.argwhere(~np.isnan(dys))[:, 0]
    xs = xs[idx]
    ys = y_pred[idx]
    dys = np.gradient(ys, xs)
    dys[np.isnan(dys)] = 0


    plt.figure(figsize=(8, 4))
    rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
    rgr.fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
    dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()
    ys_sl = np.ones(len(xs)) * np.nan
    segs = []
    for dy in np.unique(dys_dt):
        msk = dys_dt == dy
        lin_reg = LinearRegression()
        lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1))
        ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten()
        segs.append([xs[msk][0], xs[msk][-1]])
        plt.plot([xs[msk][0], xs[msk][-1]], [ys_sl[msk][0], ys_sl[msk][-1]], color='r', zorder=1)


    plt.xlabel(param, fontsize=12)
    plt.ylabel("score", fontsize=12)
    plt.scatter(xs, ys, label='raw data')
    plt.scatter(xs, ys_sl, s=12, label='segmented data', color='k', zorder=5)
    for n in range(len(segs)):
        plt.axvline(x=segs[n][0], linestyle='--', color='m')
        plt.axvline(x=segs[n][-1], linestyle='--', color='m')
    plt.legend(fontsize=12)
    plt.savefig(figname, bbox_inches='tight')

    return mape_error


def _analyze_topK(K, score, data, params):
    from scipy.stats import ks_2samp, binned_statistic

    if K > score.shape[0]:
        K = score.shape[0]

    good_idx = np.argsort(score)[::-1][:K]
    good_data = data[good_idx]
    good_scores = score[good_idx]

    rest_idx = np.argsort(score)[::-1][K:]
    rest_data = data[rest_idx]

    N = 3 if 3 < K else K
    topK_df = pd.DataFrame()
    for i in range(N):
        topK_dict = {}
        topK_dict["score"] = good_scores[i]
        for p, d in zip(params, good_data[i]):
            topK_dict[p] = d
        topK_df = pd.concat((topK_df, pd.DataFrame(topK_dict,index=[0])))

    p_values = {}
    p_values["score"] = "p_value"
    for i,p in enumerate(params):
        data_bins, data_edges = np.histogram(data[:,i], bins=10)
        good_bin_counts, _, good_bin_id = binned_statistic(
            good_data[:, i],
            np.ones_like(good_data[:, i]),
            statistic="count",
            bins=data_edges)

        rest_bin_counts, _, rest_bin_id = binned_statistic(
            rest_data[:, i],
            np.ones_like(rest_data[:, i]),
            statistic="count",
            bins=data_edges)

        # s1, p1 = ttest_ind(X_pos[:,j], X_neg[:, j], equal_var=False)
        s2, p2 = ks_2samp(good_bin_id, rest_bin_id, alternative='two-sided')  # this is better
        p_values[p] = p2

    topK_df = pd.concat((topK_df, pd.DataFrame(p_values,index=[0])))
    return topK_df



def qeRun(args):

    dump_folder_path = os.path.join(
        QE_DATA_PATH,
        datetime.now().strftime("%y-%m-%d-%H-%M-%S"))

    if not os.path.exists(dump_folder_path):
        os.makedirs(dump_folder_path)

    base_df = pd.read_csv(args.base)
    assert base_df["name"].item() in ["iperf","sysbench","fio"]
    base = base_df.to_dict()
    base["x"] = [float(i) for i in base["x"][0].split(",")]
    base["y"] = float(base["y"][0])
    base["param"] = [p for p in base["param"][0].split(",")]

    data, score, params = _loadData(args.data_path, base)

    try:
        assert data.shape[0] == score.shape[0]
    except AssertionError:
        l = min([data.shape[0], score.shape[0]])
        data = data[:l]
        score = score[:l]

    non_zero_indice = list(np.where(score!=0)[0])
    data = data[non_zero_indice]
    score = score[non_zero_indice]

    mape_errors = {}
    mape_errors["score"] = "sensi"
    for i, p in enumerate(params):
        X = copy(data[:, i])
        y = copy(score)
        figname = os.path.join(dump_folder_path, f"{p}_segment.png")
        mape_error = _segment(X=X, y=y, param=p, figname=figname, n_seg=5)
        mape_errors[p] = mape_error

    # mape_error_sum = sum(mape_errors.values())
    # for p in params:
    #     mape_errors[p] = mape_errors[p]/mape_error_sum
    mape_error_df = pd.DataFrame(mape_errors,index=[0])

    K = 10
    topK_df = _analyze_topK(K,score,data,params)
    result_df = pd.concat((topK_df, mape_error_df))
    result_df.reset_index(inplace=True)
    result_df.drop(columns=["index"], inplace=True)

    result_file = os.path.join(dump_folder_path, f"result_df.csv")
    result_df.to_csv(result_file, index=False)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('--base', help='specify the test case')
    parser.add_argument('--data_path', help='specify the test case')

    args = parser.parse_args()

    args.base = os.path.join(QE_DATA_PATH, "sysbench_base.csv")

    # args.case = "iperf"
    # args.data_path = "data/iperf_tuning/iperf_sysctl_tune"

    # args.case = "fio"
    # args.data_path = "data/fio_tuning/fio_sysctl_tune"

    # args.case = "sysbench"
    args.data_path = "data/sysbench_tuning/sysbench_sysctl_tune"

    # args.case = "ingress_sysctl"
    # args.data_dir = "mysql_ingress_sysctl_tuning/sys_tpe_ingress"

    # iperf_base = {
    #     "name": "iperf",
    #     "x": ",".join(str(i) for i in [10, 131072, 10240]),
    #     "y": str(7369999872.000),
    #     "param": "Parallel,length_buffers,window_size"}
    #
    # sysbench_base = {
    #     "name": "sysbench",
    #     "x": ",".join(str(i) for i in [1, 32768, 3, 100000]),
    #     "y": str(4357.130),
    #     "param": "threads,thread-stack-size,tables,table-size"}
    #
    # fio_base = {
    #     "name": "fit",
    #     "x": ",".join(str(i) for i in [1, 0, 8]),
    #     "y": str(81791.0),
    #     "param": "iodepth,bs,numjobs"}
    #
    # iperf_df = pd.DataFrame(iperf_base,index=[0])
    # sysbench_df = pd.DataFrame(sysbench_base,index=[0])
    # fio_df = pd.DataFrame(fio_base,index=[0])
    #
    # iperf_df.to_csv(os.path.join(QE_DATA_PATH, "iperf_base.csv"),index=False)
    # sysbench_df.to_csv(os.path.join(QE_DATA_PATH, "sysbench_base.csv"),index=False)
    # fio_df.to_csv(os.path.join(QE_DATA_PATH, "fio_base.csv"),index=False)
    #
    # iperf_df = pd.read_csv(os.path.join(QE_DATA_PATH, "iperf_base.csv"))
    # sysbench_df = pd.read_csv(os.path.join(QE_DATA_PATH, "sysbench_base.csv"))
    # fio_df = pd.read_csv(os.path.join(QE_DATA_PATH, "fio_base.csv"))

    qeRun(args)
