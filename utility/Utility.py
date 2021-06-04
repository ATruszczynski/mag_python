import random
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil, exp, sqrt

from ann_point.AnnPoint2 import *
from ann_point.HyperparameterRange import *
from ann_point.AnnPoint import *
from ann_point.Functions import *

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

def generate_population(hrange: HyperparameterRange, count: int, input_size: int, output_size: int) -> [AnnPoint2]:
    result = []
    # TODO stabilise names
    for i in range(count):
        layer_count = random.randint(hrange.hiddenLayerCountMin + 2, hrange.hiddenLayerCountMax + 2)
        layers = [[-1, input_size, None]]
        for i in range(1, layer_count):
            layers.append(generate_layer(hrange))
        layers[-1][1] = output_size
        loss_fun = hrange.lossFunSet[random.randint(0, len(hrange.lossFunSet) - 1)]
        learning_rate = random.uniform(hrange.learningRateMin, hrange.learningRateMax)
        mom_coeff = random.uniform(hrange.momentumCoeffMin, hrange.momentumCoeffMax)
        batch_size = random.uniform(hrange.batchSizeMin, hrange.batchSizeMax)

        result.append(point_from_layers(layers=layers, lossFun=loss_fun, learningRate=learning_rate, momCoeff=mom_coeff, batchSize=batch_size))

    return result

    # result = []
    # # TODO stabilise names
    # for i in range(count):
    #     hidden_layer_count = random.randint(hrange.layerCountMin, hrange.layerCountMax)
    #
    #     neuron_counts = [input_size]
    #     for i in range(hidden_layer_count):
    #         neuron_counts.append(random.randint(hrange.neuronCountMin, hrange.neuronCountMax))
    #     neuron_counts.append(output_size)
    #
    #     hidden_neuron_counts = []
    #
    #     for i in range(1, len(neuron_counts) - 1):
    #         hidden_neuron_counts.append(neuron_counts[i])
    #
    #     actFuns = []
    #     biases = []
    #     weights = []
    #
    #     for i in range(1, len(neuron_counts)):
    #         actFuns.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)])
    #         biases.append(np.random.uniform(-hrange.biaAbs, hrange.biaAbs, size=(neuron_counts[i], 1)))
    #         weights.append(np.random.uniform(-hrange.weiAbs, hrange.weiAbs, size=(neuron_counts[i], neuron_counts[i - 1])))
    #
    #     result.append(AnnPoint2(input_size=input_size, output_size=output_size, hiddenNeuronCounts=hidden_neuron_counts, activationFuns=actFuns, weights=weights, biases=biases))


def generate_layer(hrange: HyperparameterRange) -> [int, int, ActFun]:
    layer = [-1]
    layer.append(random.randint(hrange.neuronCountMin, hrange.neuronCountMax))
    layer.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)].copy())

    return layer

def get_default_hrange():
    hrange = HyperparameterRange((0, 3), (2, 256), [ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()], [CrossEntropy(), QuadDiff(), MeanDiff(), ChebyshevLoss()], (-3, 0), (-3, 0), (-5, 0)) #TODO fix
    # hrange = HyperparameterRange((0, 3), (2, 256), [ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()], [MeanDiff()], (-3, 0), (-3, 0), (-5, 0)) #TODO fix
    return hrange

def punishment_function(arg: float):
    result = 0.75 / (1 + exp(-50 * arg))

    if arg > 0:
        result += 0.25

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

    hidLayScale = hrange.hiddenLayerCountMax - hrange.hiddenLayerCountMin
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

def get_Xu_matrix(shape: (int, int), var_mul: float = 1, div: float = None) -> np.ndarray:
    if div is None:
        l = shape[1]
    else:
        l = div

    result = np.random.normal(0, var_mul / sqrt(l), shape)

    return result

def test_numpy_deep_copy():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrixcp = matrix.copy()

    assert np.array_equal(matrix, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    matrix[2, 1] = 80
    matrix[1, 1] = 50
    matrix[1, 0] = 40

    assert np.array_equal(matrix, np.array([[1, 2, 3], [40, 50, 6], [7, 80, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    matrixcp[0, 1] = 20
    matrixcp[2, 0] = 70

    assert np.array_equal(matrix, np.array([[1, 2, 3], [40, 50, 6], [7, 80, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 20, 3], [4, 5, 6], [70, 8, 9]]))






