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
            frame['ff'] = frame['ff'].abs()
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

    if parameter_name == "ff":
        maxff = df_mean.max()
        if 0.3 <= maxff <= 1:
            plt.ylim((0, 1))
        elif maxff <= 0.3:
            plt.yscale("log")
        elif maxff >= 1000:
            plt.ylim((1, 1e4))
            plt.yscale("log")
        elif maxff >= 10000:
            plt.ylim((1, 1e6))
            plt.yscale("log")
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



# dir_name = "iris_co1_ff1_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")

# dir_name = "iris_co1_ff1_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")


# dir_name = "iris_co3_ff1_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")

# dir_name = "iris_co3_ff1_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")


# dir_name = "iris_co1_ff4_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
# dir_name = "iris_co1_ff4_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
#
# dir_name = "iris_co3_ff4_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
# dir_name = "iris_co3_ff4_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")


# dir_name = "iris_co1_ff5_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
# dir_name = "iris_co1_ff5_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
#
# dir_name = "iris_co3_ff5_so"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
# dir_name = "iris_co3_ff5_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")

# dir_name = "iris_co3_ff4_sos"
# dfs =  read_all_frames_from_directory(rf"final_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
# dir_name = "iris_co3_ff4_sos_2"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
#
#
# dir_name = "iris_co3_ff4_sos_3"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")


# dir_name = "german_ff7"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
# # plot_min_max_avg(dfs, "tp", f"tp-{dir_name}")
# # plot_min_max_avg(dfs, "fn", f"fn-{dir_name}")
# # plot_min_max_avg(dfs, "fp", f"fp-{dir_name}")
# # plot_min_max_avg(dfs, "tn", f"tn-{dir_name}")

# dir_name = "german_ff8_es"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "sqrp", f"eff-{dir_name}")

# dir_name = "german_ff7_es"
# dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
# plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
# plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
# plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "sqrp", f"eff-{dir_name}")
# plot_min_max_avg(dfs, "tp", f"tp-{dir_name}")
# plot_min_max_avg(dfs, "fn", f"fn-{dir_name}")
# plot_min_max_avg(dfs, "fp", f"fp-{dir_name}")
# plot_min_max_avg(dfs, "tn", f"tn-{dir_name}")



dir_name = "wines_nm_1"
dfs =  read_all_frames_from_directory(rf"algo_tests\{dir_name}")
plot_min_max_avg(dfs, "nc", f"nc-{dir_name}")
plot_min_max_avg(dfs, "ff", f"ff-{dir_name}")
plot_min_max_avg(dfs, "eff", f"eff-{dir_name}")
plot_min_max_avg(dfs, "meff", f"meff-{dir_name}")
plot_min_max_avg(dfs, "ec", f"ec-{dir_name}")
plot_min_max_avg(dfs, "ni", f"ni-{dir_name}")

plot_min_max_avg(dfs, "mr", f"mr-{dir_name}")
# plot_min_max_avg(dfs, "sqrp", f"sqrp-{dir_name}")
# plot_min_max_avg(dfs, "linp", f"linp-{dir_name}")
# plot_min_max_avg(dfs, "pmp", f"pmp-{dir_name}")
# plot_min_max_avg(dfs, "cp", f"cp-{dir_name}")
# plot_min_max_avg(dfs, "dstp", f"dstp-{dir_name}")
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




