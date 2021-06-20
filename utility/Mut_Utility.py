# from evolving_classifier.EvolvingClassifier import *
import random

from ann_point.AnnPoint2 import *
from neural_network.ChaosNet import ChaosNet
from utility.Utility import choose_without_repetition, get_Xu_matrix, AnnPoint, point_from_layers, generate_layer, \
    get_links
from ann_point.HyperparameterRange import HyperparameterRange

def change_neuron_count(net: ChaosNet, hrange: HyperparameterRange, demanded_hidden: int):
    current_hidden = net.hidden_count
    change_hidden = demanded_hidden - current_hidden

    result = None
    if change_hidden < 0:
        result = decrease_neuron_count(net, -change_hidden)
    elif change_hidden > 0:
        result = increase_neuron_count(net, hrange, change_hidden)

    return result

def increase_neuron_count(net: ChaosNet, hrange: HyperparameterRange, to_add: int):
    input_size = net.input_size
    output_size = net.output_size
    new_hidden_count = net.hidden_count + to_add
    new_neuron_count = input_size + new_hidden_count + output_size

    min_wei = np.min(net.weights)
    max_wei = np.max(net.weights)
    min_bia = np.min(net.bias)
    max_bia = np.max(net.bias)

    new_links = get_links(input_size, output_size, new_neuron_count)
    new_links[:net.hidden_end_index, :net.hidden_end_index] = net.links[:net.hidden_end_index, :net.hidden_end_index]
    new_links[:net.hidden_end_index, -output_size:] = net.links[:net.hidden_end_index, -output_size:]


    new_weights = np.random.uniform(min_wei, max_wei, new_links.shape)
    new_weights[:net.hidden_end_index, :net.hidden_end_index] = net.weights[:net.hidden_end_index, :net.hidden_end_index]
    new_weights[:net.hidden_end_index, -output_size:] = net.weights[:net.hidden_end_index, -output_size:]
    new_weights = np.multiply(new_weights, new_links)

    new_biases = np.random.uniform(min_bia, max_bia, (1, new_neuron_count))
    new_biases[0, :net.hidden_end_index] = net.bias[0, :net.hidden_end_index]
    new_biases[0, -output_size:] = net.bias[0, -output_size:]

    new_af = net.actFuns[:net.hidden_end_index]
    for i in range(to_add): #TODO generate sequence of afs could be a function
        new_af.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)].copy())

    new_af.extend(net.actFuns[net.hidden_end_index:])

    return ChaosNet(input_size=input_size, output_size=output_size, links=new_links, weights=new_weights,
                    biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, maxit=net.maxit)


def decrease_neuron_count(net: ChaosNet, to_remove: int):
    options = list(range(net.hidden_start_index, net.hidden_end_index))
    ind_to_remove = choose_without_repetition(options, to_remove)
    ind_to_preserve = list(range(net.neuron_count))
    for i in ind_to_remove:
        ind_to_preserve.remove(i)

    ind_to_preserve = np.array(ind_to_preserve).reshape(1, -1)

    new_links = net.links[ind_to_preserve[0, :, None], ind_to_preserve]
    new_weights = net.weights[ind_to_preserve[0, :, None], ind_to_preserve]
    new_biases = net.bias[0, ind_to_preserve]
    new_af = []
    for i in range(ind_to_preserve.shape[1]):
        new_af.append(net.actFuns[ind_to_preserve[0, i]])

    return ChaosNet(input_size=net.input_size, output_size=net.output_size, links=new_links, weights=new_weights,
                    biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, maxit=net.maxit)



















