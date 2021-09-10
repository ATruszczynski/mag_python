from sklearn import datasets
import datetime

from evolving_classifier.EvolvingClassifier import *
from TupleForTest import TupleForTest
import numpy as np
import os.path

np.seterr(all='ignore')

def run_tests(tts: [TupleForTest], directory_for_tests, power: int) -> [[LsmNetwork]]:
    resultss = []

    for ddd in range(len(tts)):
        tt = tts[ddd]
        # print(f"Test {tt.name} has started at {datetime.datetime.now()}")
        subdir_name = f"{directory_for_tests}{os.path.sep}{tt.name}"
        if not os.path.exists(subdir_name):
            os.makedirs(subdir_name)

        fpath = f"{directory_for_tests}{os.path.sep}{tt.name}{os.path.sep}{tt.name}-"

        create_test_data_file(fpath, tt)

        results = []
        bests = []
        random.seed(tt.seed)
        np.random.seed(tt.seed)

        seeds = []
        for i in range(tt.rep):
            seeds.append(random.randint(0, 10**6))

        for i in range(tt.rep):
            print(f"{tt.name} - (Test: {ddd+1}/{len(tts)}, rep: {i + 1}/{tt.rep}) - at {datetime.datetime.now()}")
            ec = EvolvingClassifier()
            ec.prepare(popSize=tt.popSize, nn_data=tt.data, seed=seeds[i], hrange=tt.hrange, ct=tt.ct, mt=tt.mt,
                       st=tt.st, fft=tt.fft, fct=tt.fct)

            net = ec.run(iterations=tt.iterations, power=power)
            ec.history.to_csv_file(fpath=fpath+f"rep_{i + 1}.csv", reg=tt.reg)
            results.append(net.copy())

            if len(tt.fft) == 1:
                tr = net.test(tt.data[2], tt.data[3])
            else:
                tr = net.test(tt.data[2], tt.data[3], tt.fft[1]())

            net_to_file(net=net, dirpath=fpath + f"best_{i + 1}", tresult=tr)
            bests.append([net.copy(), tr])

        create_summary_file(fpath, bests, tt)
        resultss.append(results)

        print(f"{tt.name} has ended at {datetime.datetime.now()}")
        print("------------------------------------------------------")

    return resultss

def create_summary_file(fpath: str, bests: [[LsmNetwork, [np.ndarray, float]]], tt: TupleForTest):
    data_file = open(fpath + "summary_file.txt", "w")
    write_test_parameters(data_file=data_file, tt=tt)

    data_file.write("\n\nSummary:\n")
    for i in range(len(bests)):
        data_file.write(f"\nTest {i + 1}:\n")
        write_down_test_results(data_file, bests[i][0], bests[i][1])

    data_file.close()

def net_to_file(net: LsmNetwork, dirpath: str, tresult: [Any]):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    file = open(dirpath + os.path.sep + "non_matrix_data.txt", "w")
    file.write(f"input_size: \n{net.input_size}\n")
    file.write(f"output_size: \n{net.output_size}\n")
    file.write(f"neuron_count: \n{net.neuron_count}\n")
    file.write(f"actFuns: \n{net.get_act_fun_string()}\n")
    file.write(f"aggrFun: \n{net.aggrFun.to_string()}\n")
    file.write(f"net_it: \n{net.net_it}\n")
    file.write(f"mutation_radius: \n{net.mutation_radius}\n")
    file.write(f"multi: \n{net.multi}\n")
    file.write(f"p_prob: \n{net.p_prob}\n")
    file.write(f"p_rad: \n{net.p_rad}\n")
    file.write(f"c_prob: \n{net.c_prob}\n")
    file.write(f"swap_prob: \n{net.swap_prob}\n")

    file.write(f"results:\n")
    write_down_test_results(file, net, tresult)

    file.close()

    np.savetxt(dirpath + os.path.sep + "links.csv", net.links, delimiter=",")
    np.savetxt(dirpath + os.path.sep + "weights.csv", net.weights, delimiter=",")
    np.savetxt(dirpath + os.path.sep + "biases.csv", net.biases, delimiter=",")

def write_down_test_results(data_file, net: LsmNetwork, tr: [Any]):
    data_file.write(f"{net.to_string()}\n")
    data_file.write(f"cm:\n{tr[0]}\n")
    data_file.write(f"acc: {accuracy(tr[0])}\n")
    data_file.write(f"av_prec: {average_precision(tr[0])}\n")
    data_file.write(f"av_rec: {average_recall(tr[0])}\n")
    data_file.write(f"av_f1: {average_f1_score(tr[0])}\n")
    data_file.write(f"eff: {efficiency(tr[0])}\n")
    data_file.write(f"meff: {m_efficiency(tr[0])}\n")
    if len(tr) >= 2:
        data_file.write(f"err: {tr[1]}\n")


def write_test_parameters(data_file, tt:TupleForTest):
    data_file.write(f"test-data:\n")

    data_file.write(f"name: {tt.name} \n")
    data_file.write(f"rep: {tt.rep} \n")
    data_file.write(f"seed: {tt.seed} \n")
    data_file.write(f"popSize: {tt.popSize} \n")
    data_file.write(f"iterations: {tt.iterations} \n")
    data_file.write(f"ct: {tt.ct.__name__} \n")
    data_file.write(f"mt: {tt.mt.__name__} \n")
    data_file.write(f"st: {tt.st[0].__name__} \n")
    if len(tt.st) == 2:
        data_file.write(f"starg: {tt.st[1]} \n")
    data_file.write(f"fft: {tt.fft[0].__name__} \n")
    if len(tt.fft) == 2:
        data_file.write(f"fftarg: {tt.fft[1].__name__} \n")
    data_file.write(f"fct: {tt.fct.__name__} \n")
    data_file.write(f"reg: {tt.reg} \n")
    data_file.write(f"data_len: {len(tt.data[0])} \n")

    data_file.write(f"\nhyperparameters:\n")

    hrange = tt.hrange
    data_file.write(f"min_init_wei: {hrange.min_init_wei}\n")
    data_file.write(f"max_init_wei: {hrange.max_init_wei}\n")
    data_file.write(f"min_init_bia: {hrange.min_init_bia}\n")
    data_file.write(f"max_init_bia: {hrange.max_init_bia}\n")
    data_file.write(f"min_it: {hrange.min_it}\n")
    data_file.write(f"max_it: {hrange.max_it}\n")
    data_file.write(f"min_hidden: {hrange.min_hidden}\n")
    data_file.write(f"max_hidden: {hrange.max_hidden}\n")

    data_file.write(f"min_mut_radius: {hrange.min_mut_radius}\n")
    data_file.write(f"max_mut_radius: {hrange.max_mut_radius}\n")
    data_file.write(f"min_multi: {hrange.min_multi}\n")
    data_file.write(f"max_multi: {hrange.max_multi}\n")
    data_file.write(f"min_p_prob: {hrange.min_p_prob}\n")
    data_file.write(f"max_p_prob: {hrange.max_p_prob}\n")
    data_file.write(f"min_p_rad: {hrange.min_p_rad}\n")
    data_file.write(f"max_p_rad: {hrange.max_p_rad}\n")
    data_file.write(f"min_c_prob: {hrange.min_c_prob}\n")
    data_file.write(f"max_c_prob: {hrange.max_c_prob}\n")
    data_file.write(f"min_swap: {hrange.min_swap}\n")
    data_file.write(f"max_swap: {hrange.max_swap}\n")

    data_file.write("actfuns: ")
    for i in range(len(hrange.actFunSet)):
        data_file.write(hrange.actFunSet[i].to_string() + ", ")
    data_file.write("\n")
    if hrange.aggrFuns is not None:
        data_file.write("aggrfuns: ")
        for i in range(len(hrange.aggrFuns)):
            data_file.write(hrange.aggrFuns[i].to_string() + ", ")
        data_file.write("\n")


def create_test_data_file(fpath: str, tt: TupleForTest):
    data_file = open(fpath + "data_file.txt", "w")
    write_test_parameters(data_file, tt)

    data_file.close()




