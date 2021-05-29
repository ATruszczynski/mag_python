# from evolving_classifier.EvolvingClassifier import *
import random

from ann_point.AnnPoint2 import *
from utility.Utility import choose_without_repetition, get_Xu_matrix
from ann_point.HyperparameterRange import HyperparameterRange


def change_amount_of_layers(point: AnnPoint2, demanded: int, hrange: HyperparameterRange) -> AnnPoint2:
    result = point.copy()

    curr = len(point.hidden_neuron_counts)
    if demanded < curr:
        result = decrease_amount_of_layers(point, demanded)
    else:
        result = increase_amount_of_layers(point, demanded, hrange)

    return result

def decrease_amount_of_layers(point: AnnPoint2, demanded: int) -> AnnPoint2:
    curr = len(point.hidden_neuron_counts)
    number_to_remove = curr - demanded

    layers = point.into_numbered_layer_tuples()

    choices = list(range(1, len(layers) - 1))
    to_remove = choose_without_repetition(choices, number_to_remove)

    new_layers = []
    for i in range(len(layers)):
        if i not in to_remove:
            new_layers.append(layers[i])
    for i in range(1, len(new_layers)):
        prev_lay = new_layers[i - 1]
        cur_lay = new_layers[i]
        if prev_lay[0] != cur_lay[0] - 1:
            cur_lay[3] = get_Xu_matrix((cur_lay[1], prev_lay[1]))
            cur_lay[4] = np.zeros((cur_lay[1], 1))

    return point_from_layer_tuples(new_layers)

def increase_amount_of_layers(point: AnnPoint2, demanded: int, hrange: HyperparameterRange) -> AnnPoint2:
    curr = len(point.hidden_neuron_counts)
    number_to_add = demanded - curr

    layers = point.into_numbered_layer_tuples()

    new_layers = []
    for i in range(number_to_add):
        new_layer = [-1]
        new_layer.append(random.randint(hrange.neuronCountMin, hrange.neuronCountMax))#TODO ktoś musi sprawdzać czy w ogóle można dodać warstwy
        new_layer.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)])
        new_layers.append(new_layer)

    for i in range(len(new_layers)):
        pos = choose_without_repetition(list(range(len(layers) - 1)), 1)[0]
        layers.insert(pos + 1, new_layers[i])

    # for i in range(1, len(layers)):
    #     layer = layers[i]
    #     pre_layer = layers[i - 1]
    #     if len(layer) < 4:
    #         layer.append(get_Xu_matrix((layer[1], pre_layer[1])))
    #         layer.append(np.zeros((layer[1], 1)))
    #     elif pre_layer[1] != layer[3].shape[1] or pre_layer[0] != layer[0] - 1:
    #         layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
    #         layer[4] = np.zeros((layer[1], 1))
    #     layers[i] = layer

    layers = fix_layer_sizes(layers)

    return point_from_layer_tuples(layers)

def change_neuron_count_in_layer(point: AnnPoint2, layer: int, demanded: int) -> AnnPoint2:
    curr = point.hidden_neuron_counts[layer]
    result = point.copy()

    if demanded < curr:
        result = decrease_neuron_count_in_layer(point, layer, curr - demanded)
    elif demanded > curr:
        result = increase_neuron_count_in_layer(point, layer, demanded - curr)

    return result

def decrease_neuron_count_in_layer(point: AnnPoint2, layer: int, remove_count: int) -> AnnPoint2:
    curr = point.hidden_neuron_counts[layer]

    to_remove = choose_without_repetition(list(range(0, curr)), remove_count)
    layers = point.into_numbered_layer_tuples()

    layer_decreased = layers[layer + 1]
    next_layer_decreased = layers[layer + 2]
    layer_decreased[1] -= remove_count
    layer_decreased[3] = np.delete(layer_decreased[3], to_remove, axis=0)
    layer_decreased[4] = np.delete(layer_decreased[4], to_remove, axis=0)
    next_layer_decreased[3] = np.delete(next_layer_decreased[3], to_remove, axis=1)

    return point_from_layer_tuples(layers)

def increase_neuron_count_in_layer(point: AnnPoint2, layer: int, add_count: int) -> AnnPoint2:
    layers = point.into_numbered_layer_tuples()

    curr_layer = layers[layer + 1]
    next_layer = layers[layer + 2]

    wei_rows_to_add = get_Xu_matrix((add_count, curr_layer[3].shape[1]))
    bias_rows_to_add = np.zeros((add_count, 1))
    curr_layer[3] = np.vstack([curr_layer[3], wei_rows_to_add])
    curr_layer[4] = np.vstack([curr_layer[4], bias_rows_to_add])

    wei_cols_to_add = get_Xu_matrix((next_layer[1], add_count))
    next_layer[3] = np.hstack([next_layer[3], wei_cols_to_add])

    return point_from_layer_tuples(layers)

def fix_layer_sizes(layers: [[int, int, ActFun, np.ndarray, np.ndarray]]) -> [[int, int, ActFun, np.ndarray, np.ndarray]]:
    for i in range(1, len(layers)):
        layer = layers[i]
        pre_layer = layers[i - 1]
        if len(layer) < 4:
            layer.append(get_Xu_matrix((layer[1], pre_layer[1])))
            layer.append(np.zeros((layer[1], 1)))
        elif pre_layer[1] != layer[3].shape[1] or pre_layer[0] != layer[0] - 1:
            layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
            layer[4] = np.zeros((layer[1], 1))
        layers[i] = layer

    return layers



