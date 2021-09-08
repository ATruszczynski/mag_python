from math import log10
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_path = f"img"

# TODO - A - add saving pictures

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



# IRISES 50

dir_name = "iris_500_200_meff"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="iris_50_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="iris_50_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="iris_50_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="iris_50_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="iris_50_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Iris/50/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="iris_50_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")




dir_name = "iris_500_200_mixmeff"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="iris_50_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="iris_50_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="iris_50_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="iris_50_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="iris_50_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="iris_50_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu Iris/50/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", lim=[0, 0.4], to_draw=[True, True, False, True], spath="iris_50_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")


# IRISES 15

dir_name = "iris_500_200_15_meff"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="iris_15_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="iris_15_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="iris_15_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="iris_15_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="iris_15_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Iris/15/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="iris_15_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")




dir_name = "iris_500_200_15_mixff"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="iris_15_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="iris_15_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="iris_15_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="iris_15_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="iris_15_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="iris_15_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu Iris/15/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", lim=[0, 0.4], to_draw=[True, True, False, True], spath="iris_15_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")

# WINES

dir_name = "wwwwines_EFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="wines_100_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="wines_100_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="wines_100_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="wines_100_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="wines_100_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Wines/100/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="wines_100_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")




dir_name = "wwwwines_MIXFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="wines_100_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="wines_100_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="wines_100_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="wines_100_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="wines_100_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="wines_100_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu Wines/100/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", lim=[0.085, 0.15], to_draw=[True, False, False, True], spath="wines_100_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")




# WINES 25


dir_name = "wines_25_EFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="wines_25_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="wines_25_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="wines_25_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="wines_25_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="wines_25_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Wines/25/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="wines_25_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")

dir_name = "wines_25_MIXFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="wines_25_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="wines_25_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="wines_25_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="wines_25_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="wines_25_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="wines_25_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu Wines/25/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", lim=[0.085, 0.15], to_draw=[True, False, False, True], spath="wines_25_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")




# GERMAN 25


dir_name = "german_EFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="german_25_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="german_25_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="german_25_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="german_25_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="german_25_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu German/25/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="german_25_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")
plot_fp_fn_avg(frames=dfs, title="fn i fp dla testu German/25/Eff", xtitle="iteracja",
               ytitle="odsetek", lim=[-0.01, 1.01], spath="german_25_eff_fpnr")

dir_name = "german_MIXFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="german_25_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="german_25_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="german_25_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="german_25_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="german_25_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="german_25_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu German/25/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", to_draw=[True, False, False, True], spath="german_25_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")
plot_fp_fn_avg(frames=dfs, title="fn i fp dla testu German/25/MixFf", xtitle="iteracja",
               ytitle="odsetek", lim=[-0.01, 1.01], spath="german_25_mixff_fpnr")


# GERMAN 100


dir_name = "german_100_EFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="german_100_eff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="german_100_eff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="german_100_eff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="german_100_eff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="german_100_eff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu German/100/Eff", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="german_100_eff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
# plot_min_max_avg(frames=dfs, parameter_name="ff1", title="ff1", lim=[0.1, 0.13])
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")
plot_fp_fn_avg(frames=dfs, title="fn i fp dla testu German/100/Eff", xtitle="iteracja",
               ytitle="odsetek", lim=[-0.01, 1.01], spath="german_100_eff_fpnr")

dir_name = "german_100_MIXFF"
dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")

plot_min_max_avg(frames=dfs, parameter_name="nc", title="Liczba neuronów dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="liczba neuronów", to_draw=[True, True, True, True], spath="german_100_mixff_nc")
# plot_min_max_avg(frames=dfs, parameter_name="ec", title="ec")
plot_min_max_avg(frames=dfs, parameter_name="ni", title="Liczba iteracji sieci dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="liczba iteracji sieci", to_draw=[True, True, True, True], spath="german_100_mixff_ni")
plot_min_max_avg(frames=dfs, parameter_name="mr", title="mr dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="mr (log10)", to_draw=[True, True, True, True], spath="german_100_mixff_mr")
plot_min_max_avg(frames=dfs, parameter_name="mult", title="md dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="md (log10)", to_draw=[True, True, True, True], spath="german_100_mixff_mult")
# # plot_min_max_avg(frames=dfs, parameter_name="ppm", title="ppm")
# plot_min_max_avg(frames=dfs, parameter_name="prad", title="prad")
# plot_min_max_avg(frames=dfs, parameter_name="cp", title="cp")
# plot_min_max_avg(frames=dfs, parameter_name="swp", title="swp")
# plot_min_max_avg(frames=dfs, parameter_name="eff", title="eff", lim=[-0.01, 1.01])
plot_min_max_avg(frames=dfs, parameter_name="meff", title="Efektywność dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="efektywność", lim=[-0.01, 1.01], to_draw=[True, True, True, False], spath="german_100_mixff_meff")
plot_min_max_avg(frames=dfs, parameter_name="acc", title="accuracy dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="accuracy", lim=[-0.01, 1.01], to_draw=[True, True, True, True], spath="german_100_mixff_acc")
# plot_min_max_avg(frames=dfs, parameter_name="prc", title="prc")
# plot_min_max_avg(frames=dfs, parameter_name="rec", title="rec")
# plot_min_max_avg(frames=dfs, parameter_name="f1s", title="f1s")
plot_min_max_avg(frames=dfs, parameter_name="ff1", title="Mieszana efektywność dla testu German/100/MixFf", xtitle="iteracja",
                 ytitle="mieszana efektywność", to_draw=[True, False, False, True], spath="german_100_mixff_ff1")
# plot_min_max_avg(frames=dfs, parameter_name="ff2", title="ff2")
# plot_min_max_avg(frames=dfs, parameter_name="ff3", title="ff3")
plot_fp_fn_avg(frames=dfs, title="fn i fp dla testu German/100/MixFf", xtitle="iteracja",
               ytitle="odsetek", lim=[-0.01, 1.01], spath="german_100_mixff_fpnr")



plt.show()




