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
        plt.ylim((1, 1e6))
        plt.yscale("log")
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



# frame = read_data_from_file(r"C:\Users\aleks\Desktop\mag_python\algo_tests\iris_sos_ff4\iris_sos_ff4-rep_1.csv")
#
# print(frame.columns)
#
#
# means = frame.groupby("it").mean()
# max = frame.groupby("it").max()
#
# print(means)
# print(max)

# pats = [r"C:\Users\aleks\Desktop\mag_python\algo_tests\iris_sos_ff4\iris_sos_ff4-rep_1.csv",
#         r"C:\Users\aleks\Desktop\mag_python\algo_tests\iris_sos_ff4\iris_sos_ff4-rep_2.csv"]
#
# plot_min_max_avg(data_paths=pats, parameter_name="meff")

dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co1_ff1")
plot_min_max_avg(dfs, "nc", "nc-sos5_count_co1_ff1")

dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co1_ff1")
plot_min_max_avg(dfs, "eff", "eff-sos5_count_co1_ff1")



dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co1_ff4")
plot_min_max_avg(dfs, "nc", "nc-sos5_count_co1_ff4")

dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co1_ff4")
plot_min_max_avg(dfs, "eff", "eff-sos5_count_co1_ff4")



dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co3_ff1")
plot_min_max_avg(dfs, "nc", "nc-sos5_count_co3_ff1")

dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co3_ff1")
plot_min_max_avg(dfs, "eff", "eff-sos5_count_co3_ff1")



dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co3_ff4")
plot_min_max_avg(dfs, "nc", "nc-sos5_count_co3_ff4")

dfs =  read_all_frames_from_directory(r"algo_tests\sos5_count_co3_ff4")
plot_min_max_avg(dfs, "eff", "eff-sos5_count_co3_ff4")




