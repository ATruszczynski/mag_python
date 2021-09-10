from math import log10
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_path = f"img"

def plot_fp_fn_avg(frames: [pd.DataFrame],
                     title: str = None,
                     xtitle: str = None,
                     ytitle: str = None,
                     lim: [Any] = None,
                     scale: str = None,
                     legend_loc: int = -1,
                     spath: str = None
                     ):


    fn = []
    fp = []
    with pd.option_context('mode.use_inf_as_na', True):
        for i in range(len(frames)):
            frame = frames[i]
            frame['ff1'] = frame['ff1'].abs()
            if "ff2" in frame.columns:
                frame['ff2'] = frame['ff2'].abs()
            if "ff3" in frame.columns:
                frame['ff3'] = frame['ff3'].abs()
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            fn.append(frame.groupby("it").first()["fn"])
            fp.append(frame.groupby("it").first()["fp"])

    framess = [fn, fp]
    for i in range(len(framess)):
        frames = framess[i]
        max_len = max(f.shape[0] for f in frames)
        for j in range(len(frames)):
            frame = frames[j]
            to_add = max_len - frame.shape[0]
            frames[j] = frame.append(frame.iloc[[-1] * to_add], ignore_index=True)



    df_fn = pd.concat(fn, axis=1).mean(axis=1)
    df_fp = pd.concat(fp, axis=1).mean(axis=1)

    ax = plt.gca()


    df_fn.plot(kind='line',x='name',y='mean', color='blue', ax=ax, label="fn", title=title)
    df_fp.plot(kind='line',x='name',y='min', color="red", ax=ax, label="fp", title=title)



    if lim is not None:
        plt.ylim(lim)

    if scale is not None:
        plt.yscale(scale)

    if legend_loc >= 0:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle, fontsize=15)

    if ytitle is not None:
        plt.ylabel(ytitle, fontsize=15)

    plt.title(title, fontdict={'fontsize': 16})

    if spath is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}{os.path.sep}{spath}.png")
    plt.show()

def plot_min_max_avg(frames: [pd.DataFrame],
                     parameter_name: str,
                     title: str = None,
                     xtitle: str = None,
                     ytitle: str = None,
                     lim: [Any] = None,
                     scale: str = None,
                     legend_loc: int = -1,
                     to_draw: [bool] = None,
                     spath: str = None
                     ):


    mins = []
    means = []
    maxs = []
    firsts = []
    with pd.option_context('mode.use_inf_as_na', True):
        for i in range(len(frames)):
            frame = frames[i]
            frame['ff1'] = frame['ff1'].abs()
            if "ff2" in frame.columns:
                frame['ff2'] = frame['ff2'].abs()
            if "ff3" in frame.columns:
                frame['ff3'] = frame['ff3'].abs()
            frame.replace([np.inf, -np.inf], np.nan, inplace=True)
            firsts.append(frame.groupby("it").first()[parameter_name])
            mins.append(frame.groupby("it").min()[parameter_name])
            means.append(frame.groupby("it").mean()[parameter_name])
            maxs.append(frame.groupby("it").max()[parameter_name])

    framess = [firsts, mins, means, maxs]
    for i in range(len(framess)):
        frames = framess[i]
        max_len = max(f.shape[0] for f in frames)
        for j in range(len(frames)):
            frame = frames[j]
            to_add = max_len - frame.shape[0]
            frames[j] = frame.append(frame.iloc[[-1] * to_add], ignore_index=True)



    df_min = pd.concat(mins, axis=1).mean(axis=1)
    df_mean = pd.concat(means, axis=1).mean(axis=1)
    df_max = pd.concat(maxs, axis=1).mean(axis=1)
    df_first = pd.concat(firsts, axis=1).mean(axis=1)

    ax = plt.gca()

    if title is None:
        title = parameter_name

    if to_draw is None:
        df_mean.plot(kind='line',x='name',y='mean', color='green', ax=ax, label="mean")
        df_min.plot(kind='line',x='name',y='min', color="blue", ax=ax, label="min", title=title)
        df_max.plot(kind='line',x='name',y='max', color='red', ax=ax, label="max")
        df_first.plot(kind='line',x='name',y='first', color='violet', ax=ax, label="first")
    else:
        if to_draw[1]:
            df_mean.plot(kind='line',x='name',y='mean', color='violet', ax=ax, label="mean", title=title)
        if to_draw[0]:
            df_min.plot(kind='line',x='name',y='min', color="blue", ax=ax, label="min", title=title)
        if to_draw[2]:
            df_max.plot(kind='line',x='name',y='max', color='red', ax=ax, label="max", title=title)
        if to_draw[3]:
            df_first.plot(kind='line',x='name',y='first', color='green', ax=ax, label="best", title=title)



    if lim is not None:
        plt.ylim(lim)

    if scale is not None:
        plt.yscale(scale)

    if legend_loc >= 0:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()

    if xtitle is not None:
        plt.xlabel(xtitle, fontsize=15)

    if ytitle is not None:
        plt.ylabel(ytitle, fontsize=15)
    else:
        plt.ylabel(parameter_name)

    plt.title(title, fontdict={'fontsize': 16})

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





