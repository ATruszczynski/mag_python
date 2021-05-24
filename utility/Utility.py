import random
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil, exp, sqrt
from ann_point.HyperparameterRange import *
from ann_point.AnnPoint import *
from ann_point.Functions import *
from neural_network.LooseNet import LooseNetwork


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

def divideIntoBatches(inputs: [np.ndarray], outputs: [np.ndarray], batchSize: int) -> [[(np.ndarray, np.ndarray)]]:
    count = len(inputs)
    batchCount = ceil(count/batchSize)

    batches = []

    for i in range(0, batchCount):
        batch = []
        for j in range(i * batchSize, min((i + 1) * batchSize, count)):
            batch.append((inputs[j], outputs[j]))
        batches.append(batch)

    return batches

def generate_population(hrange: HyperparameterRange, count: int, input_size: int, output_size: int) -> [LooseNetwork]:
    result = []
    # TODO stabilise names
    for _ in range(count):
        io_neurons = input_size + output_size
        neuron_count = io_neurons + hrange.neuronCount
        # TODO outputy mogą być połączone krawędziami
        # TODO jeśli tester osobno to można by też próbować regresję

        links = np.zeros((neuron_count, neuron_count))

        tuples = []
        for i in range(input_size, neuron_count):
            for j in range(0, min(i, neuron_count - output_size + 1)):
                tuples.append((j, i))

        to_link = choose_without_repetition(tuples, random.randint(0, len(tuples) - 1))

        for tl in to_link:
            links[tl[0], tl[1]] = 1

        weights = np.zeros(links.shape)
        for i in range(input_size, neuron_count):
            prev = np.where(links[:, i] == 1)[0]
            pre_count = len(prev)
            for j in range(len(prev)):
                p = prev[j]
                weights[p, i] = random.gauss(0, 1 / sqrt(pre_count))

        biases = np.zeros((neuron_count, 1))
        actFuns = input_size * [None]
        for i in range(input_size, neuron_count):
            actFuns.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)].copy())

        net = LooseNetwork(input_size=input_size, output_size=output_size, links=links, weights=weights,
                           biases=biases, actFuns=actFuns)
        result.append(net)


    return result

def get_default_hrange():
    hrange = HyperparameterRange(neuronCount=128, actFuns=[ReLu(), Sigmoid(), TanH()])
    return hrange

def punishment_function(arg: float):
    result = 1.5 / (1 + exp(-50 * arg))

    if arg > 0:
        result += 0.5

    return result

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

def average_distance_between_points(points: [AnnPoint], hrange: HyperparameterRange) -> [float]:
    total_distance = []

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            total_distance.append(distance_between_points(points[i], points[j], hrange))

    return total_distance

def distance_between_points(pointA: AnnPoint, pointB: AnnPoint, hrange: HyperparameterRange) -> float:
    distance = 0

    hidLayScale = hrange.layerCountMax - hrange.layerCountMin
    if hidLayScale != 0:
        distance += (abs(pointA.hiddenLayerCount - pointB.hiddenLayerCount) / hidLayScale) ** 2

    neuronCountScale = hrange.neuronCountMax - hrange.neuronCountMin
    if neuronCountScale != 0:
        distance += (abs(pointA.neuronCount - pointB.neuronCount) / neuronCountScale) ** 2

    learningRateScale = hrange.learningRateMax - hrange.learningRateMin
    if learningRateScale != 0:
        distance += (abs(pointA.learningRate - pointB.learningRate) / learningRateScale) ** 2

    momentumCoeffScale = hrange.momentumCoeffMax - hrange.momentumCoeffMin
    if momentumCoeffScale != 0:
        distance += (abs(pointA.momCoeff - pointB.momCoeff) / momentumCoeffScale) ** 2

    batchSizeScale = hrange.batchSizeMax - hrange.batchSizeMin
    if batchSizeScale != 0:
        distance += (abs(pointA.batchSize - pointB.batchSize) / batchSizeScale) ** 2

    if pointA.actFun.to_string() != pointB.actFun.to_string():
        distance += 1

    if pointA.aggrFun.to_string() != pointB.aggrFun.to_string():
        distance += 1

    if pointA.lossFun.to_string() != pointB.lossFun.to_string():
        distance += 1

    return sqrt(distance)
# TODO end algo faster if net is good

def get_in_radius(current: float, min_val: float, max_val:float, radius: float) -> float:
    scale = max_val - min_val
    move_radius = radius * scale
    lower_bound = max(min_val, current - move_radius)
    upper_bound = min(max_val, current + move_radius)

    return random.uniform(lower_bound, upper_bound)


