import random
from itertools import combinations
from statistics import mean
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil, exp, sqrt, log10

from ann_point.HyperparameterRange import *
from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint
from utility.Utility2 import get_weight_mask
import pandas as pd

# TODO - B - remove needless functions
def try_choose_different(current: Any, possibilities: [Any]) -> Any:
    options = []
    for i in range(len(possibilities)):
        obj = possibilities[i]

        if isinstance(obj, int):
            if obj != current:
                options.append(i)
        else:
            if obj.to_string() != current.to_string():
                options.append(i)

    if len(options) == 0:
        result = current
    else:
        ind = random.randint(0, len(options) - 1)
        choice = options[ind]
        result = possibilities[choice]

    if not isinstance(result, int):
        result = result.copy()

    return result

def choose_without_repetition(options: [Any], count: int) -> [Any]:
    indices = list(range(0, len(options)))
    count = min(count, len(options))
    chosen = []
    for i in range(count):
        c = indices[random.randint(0, len(indices) - 1)]
        chosen.append(c)
        indices.remove(c)

    return [options[i] for i in chosen]

def one_hot_endode(data: [int]) -> [np.ndarray]:
    array = np.array(data)
    array = np.reshape(array, (-1, 1))

    onehot_encoder = OneHotEncoder(sparse=False)
    ohe = onehot_encoder.fit_transform(array)

    return [en.reshape(-1, 1) for en in ohe]

def generate_population(hrange: HyperparameterRange, count: int, input_size: int, output_size: int) -> [ChaosNet]:
    result = []
    # TODO - C - stabilise names
    for i in range(count):
        hidden_size = random.randint(hrange.min_hidden, hrange.max_hidden)

        neuron_count = input_size + hidden_size + output_size
        links = get_links(input_size, output_size, neuron_count)

        weights = np.random.uniform(hrange.min_init_wei, hrange.max_init_wei, (neuron_count, neuron_count))
        weights = np.random.normal(0, hrange.max_init_wei, (neuron_count, neuron_count))#!!!
        weights = np.multiply(weights, links)

        biases = np.random.uniform(hrange.min_init_bia, hrange.max_init_bia, (1, neuron_count))
        biases = np.random.normal(0, hrange.max_init_bia, (1, neuron_count))#!!!

        biases[0, :input_size] = 0

        actFuns = []
        for j in range(input_size):#TODO - C - shorten?
            actFuns.append(None)
        for j in range(input_size, input_size + hidden_size):
            actFuns.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)])
        for j in range(input_size + hidden_size, neuron_count):
            actFuns.append(None)

        aggrFun = hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)]
        maxit = random.randint(hrange.min_it, hrange.max_it)

        mut_radius = random.uniform(hrange.min_mut_radius, hrange.max_mut_radius)
        wb_mut_prob = random.uniform(hrange.min_sqr_mut_prob, hrange.max_sqr_mut_prob)
        s_mut_prob = random.uniform(hrange.min_lin_mut_prob, hrange.max_lin_mut_prob)
        p_mut_prob = random.uniform(hrange.min_p_mut_prob, hrange.max_p_mut_prob)
        c_prob = random.uniform(hrange.min_c_prob, hrange.max_c_prob)
        r_prob = random.uniform(hrange.min_dstr_mut_prob, hrange.max_dstr_mut_prob)
        a_prob = random.uniform(hrange.min_act_mut_prob, hrange.max_act_mut_prob)

        # mut_radius = hrange.max_mut_radius
        # wb_mut_prob = hrange.max_sqr_mut_prob
        # s_mut_prob = hrange.max_lin_mut_prob
        # p_mut_prob = hrange.max_p_mut_prob
        # c_prob = hrange.max_c_prob
        # r_prob = hrange.max_dstr_mut_prob
        # a_prob = hrange.max_act_mut_prob
        #
        mut_radius = hrange.max_mut_radius
        wb_mut_prob = hrange.min_sqr_mut_prob
        s_mut_prob = hrange.min_lin_mut_prob
        p_mut_prob = hrange.min_p_mut_prob # !!!
        c_prob = hrange.max_c_prob
        r_prob = hrange.max_dstr_mut_prob
        a_prob = hrange.max_act_mut_prob

        # mut_radius = random.uniform((hrange.min_mut_radius + hrange.max_mut_radius)/2, hrange.max_mut_radius)
        # wb_mut_prob = random.uniform(hrange.min_sqr_mut_prob, (hrange.min_sqr_mut_prob + hrange.max_sqr_mut_prob)/2)
        # s_mut_prob = random.uniform(hrange.min_lin_mut_prob, (hrange.min_lin_mut_prob + hrange.max_lin_mut_prob)/2)
        # p_mut_prob = random.uniform(hrange.min_p_mut_prob, (hrange.min_p_mut_prob + hrange.max_p_mut_prob)/2)
        # c_prob = random.uniform(hrange.min_c_prob, hrange.max_c_prob)
        # r_prob = random.uniform(hrange.min_dstr_mut_prob, hrange.max_dstr_mut_prob)
        # a_prob = random.uniform(hrange.min_act_mut_prob, hrange.max_act_mut_prob)


        cn = ChaosNet(input_size=input_size, output_size=output_size, links=links, weights=weights,
                      biases=biases, actFuns=actFuns, aggrFun=aggrFun, net_it=maxit, mutation_radius=mut_radius,
                      sqr_mut_prob=wb_mut_prob, lin_mut_prob=s_mut_prob, p_mutation_prob=p_mut_prob,
                      c_prob=c_prob, dstr_mut_prob=r_prob, act_mut_prob=a_prob)

        cndp = CNDataPoint(cn)

        cn.bun = 1
        result.append(cndp)

    return result

def get_links(input_size: int, output_size: int, neuron_count: int):
    mask = get_weight_mask(input_size, output_size, neuron_count)

    density = random.random()
    density = 1
    # density = 0.5
    link_prob = np.random.random((neuron_count, neuron_count))
    conn_ind = np.where(link_prob <= density)
    links = np.zeros((neuron_count, neuron_count))
    links[conn_ind] = 1
    # links[:input_size, input_size:-output_size] = 1
    # links[input_size:-output_size, -output_size:] = 1
    links = np.multiply(links, mask)

    return links

#TODO - B - zasadniczo możnaby wyrzucić tworzenie obiektów funkcji tutaj (done?)
def get_default_hrange_ga():#TODO - S - przemyśl to
    hrange = HyperparameterRange(init_wei=(-0.1, 0.1), init_bia=(-0.1, 0.1), it=(1, 10), hidden_count=(0, 50),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-1, -1), sqr_mut_prob=(-2, -2), lin_mut_prob=(-1, -1),
                                 p_mutation_prob=(-100, -100), c_prob=(log10(0.8), log10(0.8)),
                                 dstr_mut_prob=(log10(0.005), log10(0.005)))
    return hrange

#TODO - B - zasadniczo możnaby wyrzucić tworzenie obiektów funkcji tutaj (done?)
def get_default_hrange_ga2():#TODO - S - przemyśl to
    hrange = HyperparameterRange(init_wei=(-0.1, 0.1), init_bia=(-0.1, 0.1), it=(1, 10), hidden_count=(0, 30),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-3.5, -3.5), sqr_mut_prob=(-2.5, -2.5), lin_mut_prob=(-1.5, -1.5),
                                 p_mutation_prob=(-100, -100), c_prob=(log10(0.8), log10(0.8)),
                                 dstr_mut_prob=(log10(0.001), log10(0.001)))
    return hrange

def get_default_hrange_es():
    hrange = HyperparameterRange(init_wei=(-0.1, 0.1), init_bia=(-0.1, 0.1), it=(1, 10), hidden_count=(30, 30),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-5, -2), sqr_mut_prob=(-3, -1), lin_mut_prob=(-3, -1),
                                 p_mutation_prob=(-2, 0), c_prob=(log10(0.6), log10(1)),
                                 dstr_mut_prob=(log10(0.005), log10(0.1)), act_mut_prob=(-2, -1))
    return hrange

def get_default_hrange_es2():
    hrange = HyperparameterRange(init_wei=(-0.1, 0.1), init_bia=(-0.1, 0.1), it=(1, 10), hidden_count=(0, 30),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-5, 0), sqr_mut_prob=(-3, 0), lin_mut_prob=(-3, 0),
                                 p_mutation_prob=(-2, 0), c_prob=(log10(0.6), log10(1)),
                                 dstr_mut_prob=(-5, 0))
    return hrange

def get_default_hrange_es3():
    d = 0
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(20, 20),
                                 actFuns=[ReLu()],
                                 mut_radius=(-2, -1), sqr_mut_prob=(log10(0.05), log10(0.5)),
                                 lin_mut_prob=(log10(0.01), log10(0.1)),
                                 p_mutation_prob=(-1, -1), c_prob=(log10(0.8), log10(0.8)),
                                 dstr_mut_prob=(-100, -100), act_mut_prob=(-100, -100))

    # hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(40, 40),
    #                              actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid()],
    #                              mut_radius=(log10(d/10), log10(d)), sqr_mut_prob=(log10(0.8), log10(0.8)),
    #                              lin_mut_prob=(log10(1), log10(1)),
    #                              p_mutation_prob=(-1, -1), c_prob=(-100, -100),
    #                              dstr_mut_prob=(-2, -1), act_mut_prob=(-2, -2))

    # hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(40, 40),
    #                              actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid()],
    #                              mut_radius=(-100, -100), sqr_mut_prob=(-100, -100),
    #                              lin_mut_prob=(-100, -100),
    #                              p_mutation_prob=(-100, -100), c_prob=(-100, -100),
    #                              dstr_mut_prob=(-2, -2), act_mut_prob=(-100, -100))

    #
    # hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(40, 40),
    #                              actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid()],
    #                              mut_radius=(-100, -100), sqr_mut_prob=(-100, -100),
    #                              lin_mut_prob=(-100, -100),
    #                              p_mutation_prob=(-100, -100), c_prob=(log10(0.8), log10(0.8)),
    #                              dstr_mut_prob=(-2, -2), act_mut_prob=(-100, -100))

    # hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(40, 40),
    #                              actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid()],
    #                              mut_radius=(log10(d/10), log10(10*d)), sqr_mut_prob=(log10(0.8), log10(0.8)),
    #                              lin_mut_prob=(log10(1), log10(1)),
    #                              p_mutation_prob=(-1, -1), c_prob=(-100, -100),
    #                              dstr_mut_prob=(-2, -1), act_mut_prob=(-2, -2))

    # c_prob=(-100, -100)
    # c_prob=(log10(0.6), log10(1))
    # c_prob=(log10(0.8), log10(0.8))
    return hrange

def get_default_hrange_es4():
    d = 0
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(1, 1), hidden_count=(15, 15),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-4, -1), sqr_mut_prob=(log10(0.01), log10(1)),
                                 lin_mut_prob=(log10(0.1), log10(1)),
                                 p_mutation_prob=(-3, -1), c_prob=(log10(0.8), log10(0.8)),
                                 dstr_mut_prob=(-100, -100), act_mut_prob=(-100, -100))

    return hrange

def get_default_hrange_es5():
    d = 0.0
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=ddd, init_bia=ddd, it=(3, 3), hidden_count=(15, 15),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-4, -1), sqr_mut_prob=(log10(0.01), log10(1)),
                                 lin_mut_prob=(log10(0.1), log10(1)),
                                 p_mutation_prob=(-2, -1), c_prob=(log10(0.8), log10(0.8)),
                                 dstr_mut_prob=(-100, -100), act_mut_prob=(-100, -100))

    return hrange

def generate_counting_problem(howMany: int, countTo: int) -> [np.ndarray]:
    inputs = []
    outputs = []

    for i in range(howMany):
        c = random.randint(0, countTo)

        input = np.zeros((countTo, 1))
        output = np.zeros((countTo + 1, 1))

        to_color = choose_without_repetition(list(range(0, input.shape[0])), c)
        input[to_color, 0] = 1
        output[c, 0] = 1

        inputs.append(input)
        outputs.append(output)

    return [inputs, outputs]

def generate_counting_problem_unique(countTo: int) -> [np.ndarray]:
    inputs = []
    outputs = []

    choices = list(range(0, countTo))
    inds = []

    for i in range(len(choices) + 1):
        inds.extend(list(combinations(choices, i)))

    for i in range(len(inds)):
        inp = np.zeros((countTo, 1))
        out = np.zeros((countTo + 1, 1))

        ind = inds[i]

        inp[ind, 0] = 1
        out[len(ind), 0] = 1

        inputs.append(inp)
        outputs.append(out)

    return inputs, outputs




# TODO - B - generate unblaanced counting problem
# def generate_counting_problem(countTo: int):
#     inputs = []
#     outputs = []



def generate_square_problem(howMany, minV, maxV) -> [np.ndarray]:
    inputs = []
    outputs = []

    for i in range(howMany):
        x = random.uniform(minV, maxV)
        y = x ** 2
        # y = 2 * x

        input = np.zeros((1, 1))
        output = np.zeros((1, 1))

        input[0, 0] = x
        output[0, 0] = y

        inputs.append(input)
        outputs.append(output)

    return [inputs, outputs]

# TODO - B - end algo faster if net is good

def get_in_radius(current: float, min_val: float, max_val:float, radius: float) -> float:
    scale = max_val - min_val
    move_radius = radius * scale
    lower_bound = max(min_val, current - move_radius)
    upper_bound = min(max_val, current + move_radius)

    return random.uniform(lower_bound, upper_bound)

def compare_lists(l1: [int], l2: [int]):
    equal = True
    equal = equal and len(l1) == len(l2)

    for i in range(len(l1)):
        equal = equal and l1[i] == l2[i]

    return equal

def copy_list_of_arrays(arrays: [np.ndarray]) -> [np.ndarray]:
    results = []
    for i in range(len(arrays)):
        results.append(arrays[i].copy())

    return results



def get_testing_hrange():
    return HyperparameterRange(
        init_wei=(0, 1),
        init_bia=(2, 3),
        it=(4, 5),
        hidden_count=(6, 7),
        actFuns=[ReLu(), Identity(), Sigmoid(), Poly2()],
        mut_radius=(8, 9),
        sqr_mut_prob=(10, 11),
        lin_mut_prob=(12, 13),
        p_mutation_prob=(14, 15),
        c_prob=(16, 17),
        dstr_mut_prob=(18, 19)
    )

def translate_german(data_frame: pd.DataFrame, fpath: str):
    data_type = data_frame.dtypes
    replace_dict = {}
    for i in range(len(data_frame.columns)):
        if data_type[i] == object:
            uniques = data_frame.iloc[:, i].unique().tolist()
            uniques = sorted(uniques)
            for j in range(len(uniques)):
                replace_dict[uniques[j]] = j

    data_frame.replace(to_replace=replace_dict, inplace=True)
    data_frame = data_frame.astype(np.dtype("int64"))
    data_frame.iloc[:, [-1]] = data_frame.iloc[:, [-1]] - 1

    data_frame.to_csv(fpath, header=False, index=False)

def divide_frame_into_columns(data_frame: pd.DataFrame) -> [np.ndarray]:
    res = []
    for i in range(len(data_frame.columns)):
        res.append(data_frame.iloc[:, i].to_numpy().reshape(-1, 1))

    return res

def translate_wines(data_frame: pd.DataFrame, fpath: str):
    data_frame.iloc[:, -1] = data_frame.iloc[:, -1] - data_frame.iloc[:, -1].max()

    ori = 1









