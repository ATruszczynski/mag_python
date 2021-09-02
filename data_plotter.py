from math import log10

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_neuron_counts():
    pass

def plot_min_max_avg(frames: [pd.DataFrame], parameter_name: str, title: str):
    mins = []
    means = []
    maxs = []
    with pd.option_context('mode.use_inf_as_na', True):
        for i in range(len(frames)):
            frame = frames[i]
            frame['ff1'] = frame['ff1'].abs()
            if "ff2" in frame.index:
                frame['ff2'] = frame['ff2'].abs()
            if "ff3" in frame.index:
                frame['ff3'] = frame['ff3'].abs()
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            mins.append(frame.groupby("it").min()[parameter_name])
            means.append(frame.groupby("it").mean()[parameter_name])
            maxs.append(frame.groupby("it").max()[parameter_name])

    df_min = pd.concat(mins, axis=1).min(axis=1)
    df_mean = pd.concat(means, axis=1).mean(axis=1)
    df_max = pd.concat(maxs, axis=1).max(axis=1)

    ax = plt.gca()

    df_min.plot(kind='line',x='name',y='min', color="blue", ax=ax, label="min", title=title)
    df_mean.plot(kind='line',x='name',y='mean', color='green', ax=ax, label="mean")
    df_max.plot(kind='line',x='name',y='max', color='red', ax=ax, label="max")

    if "ff1" is parameter_name or "ff2" is parameter_name or "ff3" is parameter_name:
        maxff = df_min.max()
        if 0.2 <= maxff <= 1:
            plt.ylim((0, 1))
        elif maxff <= 0.3:
            plt.yscale("log")
        elif maxff >= 1000:
            plt.ylim((1, 1e4))
            plt.yscale("log")
        elif maxff >= 10000:
            plt.ylim((1, 1e6))
            plt.yscale("log")
        else:
            plt.ylim((0, maxff))
    if parameter_name == "eff" or parameter_name == "meff":
        plt.ylim((-0.1, 1.1))
    plt.ylabel(parameter_name)
    plt.legend()
    plt.show()

def read_data_from_file(data_path: str) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    return frame

def read_all_frames_from_directory(dir_path: str) -> [pd.DataFrame]:
    files = []
    for obj in os.listdir(dir_path):
        path_to_obj = os.path.join(dir_path, obj)
        if os.path.isfile(path_to_obj) and "rep" in path_to_obj:
            files.append(path_to_obj)

    dfs = []
    for p in files:
        dfs.append(read_data_from_file(p))

    return dfs






# dir_name = "wwiness_avmin"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "ff1", f"ff1-{dir_name}")
# # plot_min_max_avg(dfs, "ff2", f"ff2-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "meff", f"meff-{dir_name}")
# plot_min_max_avg(dfs, "ec", f"ec-{dir_name}")
# plot_min_max_avg(dfs, "ni", f"ni-{dir_name}")
# # plot_min_max_avg(dfs, "f1s", f"f1s-{dir_name}")
#
# # plot_min_max_avg(dfs, "mr", f"mr-{dir_name}")
# # plot_min_max_avg(dfs, "sqrp", f"sqrp-{dir_name}")
# # plot_min_max_avg(dfs, "linp", f"linp-{dir_name}")
# # plot_min_max_avg(dfs, "pmp", f"pmp-{dir_name}")
# # plot_min_max_avg(dfs, "cp", f"cp-{dir_name}")
# # plot_min_max_avg(dfs, "dstp", f"dstp-{dir_name}")
# # plot_min_max_avg(dfs, "afp", f"afp-{dir_name}")


dir_name = "wwiness_avmin"
dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
plot_min_max_avg(dfs, "ff1", f"ff1-{dir_name}")
# # plot_min_max_avg(dfs, "ff2", f"ff2-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "meff", f"meff-{dir_name}")
# plot_min_max_avg(dfs, "ec", f"ec-{dir_name}")
plot_min_max_avg(dfs, "acc", f"acc-{dir_name}")
# plot_min_max_avg(dfs, "ni", f"ni-{dir_name}")
# plot_min_max_avg(dfs, "f1s", f"f1s-{dir_name}")

plot_min_max_avg(dfs, "mr", f"mr-{dir_name}")
# plot_min_max_avg(dfs, "sqrp", f"sqrp-{dir_name}")
plot_min_max_avg(dfs, "linp", f"linp-{dir_name}")
# plot_min_max_avg(dfs, "pmp", f"pmp-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"cp-{dir_name}")
plot_min_max_avg(dfs, "dstp", f"dstp-{dir_name}")
# plot_min_max_avg(dfs, "afp", f"afp-{dir_name}")

# dir_name = "wines2_cp2"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"cp-{dir_name}")

# dir_name = "wines2_cp2"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "ec", f"cp-{dir_name}")

# dir_name = "wines2_co3"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"cp-{dir_name}")

# plot_min_max_avg(dfs, "tp", f"tp-{dir_name}")
# plot_min_max_avg(dfs, "fn", f"fn-{dir_name}")
# plot_min_max_avg(dfs, "fp", f"fp-{dir_name}")
# plot_min_max_avg(dfs, "tn", f"tn-{dir_name}")

plt.show()




