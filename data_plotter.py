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
    with pd.option_context('mode.use_inf_as_na', True):
        for i in range(len(frames)):
            frame = frames[i]
            frame['ff1'] = frame['ff1'].abs()
            if "ff2" in frame.columns:
                frame['ff2'] = frame['ff2'].abs()
            if "ff3" in frame.columns:
                frame['ff3'] = frame['ff3'].abs()
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            mins.append(frame.groupby("it").min()[parameter_name])
            means.append(frame.groupby("it").mean()[parameter_name])
            maxs.append(frame.groupby("it").max()[parameter_name])

    df_min = pd.concat(mins, axis=1).min(axis=1)
    df_mean = pd.concat(means, axis=1).mean(axis=1)
    df_max = pd.concat(maxs, axis=1).max(axis=1)

    ax = plt.gca()

    if title is None:
        title = parameter_name

    df_min.plot(kind='line',x='name',y='min', color="blue", ax=ax, label="min", title=title)
    df_mean.plot(kind='line',x='name',y='mean', color='green', ax=ax, label="mean")
    df_max.plot(kind='line',x='name',y='max', color='red', ax=ax, label="max")


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

dir_name = "writing test"
dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="desu2",
                 scale="log", legend_loc= 9, spath= "desuuuu")
plt.show()




