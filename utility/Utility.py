import random
from typing import Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from math import ceil
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

    result = None

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

def generate_population(hrange: HyperparameterRange, count: int, input_size: int, output_size: int) -> [AnnPoint]:
    result = []
    # TODO stabilise names
    for i in range(count):
        layer_count = random.randint(hrange.layerCountMin, hrange.layerCountMax)
        neuron_count = random.uniform(hrange.neuronCountMin, hrange.neuronCountMax)
        act_fun = hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)]
        aggr_fun = hrange.aggrFunSet[random.randint(0, len(hrange.aggrFunSet) - 1)]
        loss_fun = hrange.lossFunSet[random.randint(0, len(hrange.lossFunSet) - 1)]
        learning_rate = random.uniform(hrange.learningRateMin, hrange.learningRateMax)
        mom_coeff = random.uniform(hrange.momentumCoeffMin, hrange.momentumCoeffMax)

        result.append(AnnPoint(inputSize=input_size, outputSize=output_size, hiddenLayerCount=layer_count, neuronCount=neuron_count,
                               actFun=act_fun, aggrFun=aggr_fun, lossFun=loss_fun, learningRate=learning_rate, momCoeff=mom_coeff))

    return result

def get_default_hrange():
    hrange = HyperparameterRange((0,3), (0, 8), [ReLu(), Sigmoid()], [ReLu(), Softmax()], [QuadDiff()], (-6, 0), (-6, 0))
    return hrange
