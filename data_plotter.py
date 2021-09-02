from math import log10
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_neuron_counts():
    pass

save_path = f"img"

# TODO - A - add saving pictures
def plot_min_max_avg(frames: [pd.DataFrame],
                     parameter_name: str,
                     title: str = None,
                     xtitle: str = None,
                     ytitle: str = None,
                     lim: [Any] = None,
                     scale: str = None,
                     legend_loc: int = -1,
                     spath: str = None
                     ):


    mins = []
    means = []
    maxs = []
    first = []
    with pd.option_context('mode.use_inf_as_na', True):
        for i in range(len(frames)):
            frame = frames[i]
            frame['ff1'] = frame['ff1'].abs()
            if "ff2" in frame.columns:
                frame['ff2'] = frame['ff2'].abs()
            if "ff3" in frame.columns:
                frame['ff3'] = frame['ff3'].abs()
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            first.append(frame.groupby("it").first()[parameter_name])
            mins.append(frame.groupby("it").min()[parameter_name])
            means.append(frame.groupby("it").mean()[parameter_name])
            maxs.append(frame.groupby("it").max()[parameter_name])

    df_min = pd.concat(mins, axis=1).min(axis=1)
    df_mean = pd.concat(means, axis=1).mean(axis=1)
    df_max = pd.concat(maxs, axis=1).max(axis=1)
    df_first = pd.concat(first, axis=1).mean(axis=1)

    ax = plt.gca()

    if title is None:
        title = parameter_name

    df_min.plot(kind='line',x='name',y='min', color="blue", ax=ax, label="min", title=title)
    df_mean.plot(kind='line',x='name',y='mean', color='green', ax=ax, label="mean")
    df_max.plot(kind='line',x='name',y='max', color='red', ax=ax, label="max")
    df_first.plot(kind='line',x='name',y='first', color='violet', ax=ax, label="first")


    if lim is not None:
        plt.ylim(lim)

    if scale is not None:
        plt.yscale(scale)

    if legend_loc >= 0:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle)

    if ytitle is not None:
        plt.ylabel(ytitle)
    else:
        plt.ylabel(parameter_name)

    if spath is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}{os.path.sep}{spath}.png")
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

dir_name = "iris_doc_avmax"
dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
# plot_min_max_avg(frames=dfs, parameter_name="ni", title="ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="mult")
plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
# plot_min_max_avg(frames=dfs, parameter_name="meff", title="meff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="acc", title="acc", lim=[-0.01, 1.01])
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.15])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")

# param = "mr"
#
# dir_name1 = "AT_wines_dco_size_1"
# dfs1 =  read_all_frames_from_directory(rf"algo_tests\{dir_name1}")
# plot_min_max_avg(frames=dfs1, parameter_name=param, title=param)
#
# dir_name2 = "AT_wines_dco_size_2"
# dfs2 =  read_all_frames_from_directory(rf"algo_tests\{dir_name2}")
# plot_min_max_avg(frames=dfs2, parameter_name=param, title=param)
#
# dir_name3 = "AT_wines_dco_size_3"
# dfs3 =  read_all_frames_from_directory(rf"algo_tests\{dir_name3}")
# plot_min_max_avg(frames=dfs3, parameter_name=param, title=param)
#
# dir_name4 = "AT_wines_dco_size_4"
# dfs4 =  read_all_frames_from_directory(rf"algo_tests\{dir_name4}")
# plot_min_max_avg(frames=dfs4, parameter_name=param, title=param)

# dir_name5 = "AT_wines_dco_size_5"
# dfs5 =  read_all_frames_from_directory(rf"algo_tests\{dir_name5}")
# plot_min_max_avg(frames=dfs5, parameter_name=param, title=param)

plt.show()




