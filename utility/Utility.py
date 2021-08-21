import random
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil, exp, sqrt

from ann_point.HyperparameterRange import *
from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.Utility2 import get_weight_mask

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
        weights = np.multiply(weights, links)

        biases = np.random.uniform(hrange.min_init_bia, hrange.max_init_bia, (1, neuron_count))
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
        wb_mut_prob = random.uniform(hrange.min_wb_mut_prob, hrange.max_wb_mut_prob)
        s_mut_prob = random.uniform(hrange.min_s_mut_prob, hrange.max_s_mut_prob)
        p_mut_prob = random.uniform(hrange.min_p_mut_prob, hrange.max_p_mut_prob)
        c_prob = random.uniform(hrange.min_c_prob, hrange.max_c_prob)
        r_prob = random.uniform(hrange.min_r_prob, hrange.max_r_prob)


        result.append(ChaosNet(input_size=input_size, output_size=output_size, links=links, weights=weights,
                               biases=biases, actFuns=actFuns, aggrFun=aggrFun, maxit=maxit, mutation_radius=mut_radius,
                               wb_mutation_prob=wb_mut_prob, s_mutation_prob=s_mut_prob, p_mutation_prob=p_mut_prob,
                               c_prob=c_prob, r_prob=r_prob))

    return result

def get_links(input_size: int, output_size: int, neuron_count: int):
    mask = get_weight_mask(input_size, output_size, neuron_count)

    density = random.random()
    link_prob = np.random.random((neuron_count, neuron_count))
    conn_ind = np.where(link_prob <= density)
    links = np.zeros((neuron_count, neuron_count))
    links[conn_ind] = 1
    links = np.multiply(links, mask)

    return links

#TODO - B - zasadniczo możnaby wyrzucić tworzenie obiektów funkcji tutaj (done?)
def get_default_hrange():#TODO - S - przemyśl to
    hrange = HyperparameterRange(init_wei=(-10, 10), init_bia=(-10, 10), it=(1, 10), hidden_count=(0, 100),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(0.0, 1), wb_mut_prob=(0.001, 0.1), s_mut_prob=(0, 1),
                                 p_mutation_prob=(0.05, 0.01), c_prob=(0.2, 1),
                                 r_prob=(0, 1))
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






