import random
# from neural_network.ChaosNet import ChaosNet

from utility.Utility import *
# from utility.Utility2 import *
# from ann_point.HyperparameterRange import HyperparameterRange

# TODO - B - remove needless functions
# TODO - A - are things here tested?

def change_neuron_count(net: ChaosNet, hrange: HyperparameterRange, demanded_hidden: int):
    current_hidden = net.hidden_count
    change_hidden = demanded_hidden - current_hidden

    result = None
    if change_hidden < 0:
        result = decrease_neuron_count(net, -change_hidden)
    elif change_hidden > 0:
        result = increase_neuron_count(net, hrange, change_hidden)
    else:
        result = net.copy()

    return result

def increase_neuron_count(net: ChaosNet, hrange: HyperparameterRange, to_add: int):
    input_size = net.input_size
    output_size = net.output_size
    new_hidden_count = net.hidden_count + to_add
    new_neuron_count = input_size + new_hidden_count + output_size

    min_wei = np.min(net.weights)
    max_wei = np.max(net.weights)
    min_bia = np.min(net.biases)
    max_bia = np.max(net.biases)

    new_links = get_links(input_size, output_size, new_neuron_count)
    new_links[:net.hidden_end_index, :net.hidden_end_index] = net.links[:net.hidden_end_index, :net.hidden_end_index]
    new_links[:net.hidden_end_index, -output_size:] = net.links[:net.hidden_end_index, -output_size:]
    new_links = np.multiply(new_links, get_weight_mask(input_size=input_size, output_size=output_size,
                                                       neuron_count=new_neuron_count))

    new_weights = np.random.uniform(min_wei, max_wei, new_links.shape)
    new_weights[:net.hidden_end_index, :net.hidden_end_index] = net.weights[:net.hidden_end_index, :net.hidden_end_index]
    new_weights[:net.hidden_end_index, -output_size:] = net.weights[:net.hidden_end_index, -output_size:]
    new_weights = np.multiply(new_weights, new_links)

    new_biases = np.random.uniform(min_bia, max_bia, (1, new_neuron_count))
    new_biases[0, :net.hidden_end_index] = net.biases[0, :net.hidden_end_index]
    new_biases[0, -output_size:] = net.biases[0, -output_size:]

    new_af = net.actFuns[:net.hidden_end_index]
    for i in range(to_add): #TODO - C - generate sequence of afs could be a function
        new_af.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)].copy())

    new_af.extend(net.actFuns[net.hidden_end_index:])

    return ChaosNet(input_size=input_size, output_size=output_size, links=new_links, weights=new_weights,
                    biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, net_it=net.net_it, mutation_radius=net.mutation_radius,
                    sqr_mut_prob=net.sqr_mut_prob, lin_mut_prob=net.lin_mut_prob, p_mutation_prob=net.p_mutation_prob,
                    c_prob=net.c_prob, dstr_mut_prob=net.dstr_mut_prob)


def decrease_neuron_count(net: ChaosNet, to_remove: int):
    options = list(range(net.hidden_start_index, net.hidden_end_index))
    ind_to_remove = choose_without_repetition(options, to_remove)
    ind_to_preserve = list(range(net.neuron_count))
    for i in ind_to_remove:
        ind_to_preserve.remove(i)

    ind_to_preserve = np.array(ind_to_preserve).reshape(1, -1)

    new_links = net.links[ind_to_preserve[0, :, None], ind_to_preserve]
    new_weights = net.weights[ind_to_preserve[0, :, None], ind_to_preserve]
    new_biases = net.biases[0, ind_to_preserve]
    new_af = []
    for i in range(ind_to_preserve.shape[1]):
        new_af.append(net.actFuns[ind_to_preserve[0, i]])

    return ChaosNet(input_size=net.input_size, output_size=net.output_size, links=new_links, weights=new_weights,
                    biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, net_it=net.net_it, mutation_radius=net.mutation_radius,
                    sqr_mut_prob=net.sqr_mut_prob, lin_mut_prob=net.lin_mut_prob, p_mutation_prob=net.p_mutation_prob,
                    c_prob=net.c_prob, dstr_mut_prob=net.dstr_mut_prob)

# def inflate_network(net: ChaosNet, to_add: int): #TODO - D - tests missed wrong maxit
#     new_neuron_count = net.neuron_count + to_add
#
#     new_links = np.zeros((new_neuron_count, new_neuron_count))
#     new_weights = np.zeros((new_neuron_count, new_neuron_count))
#     new_biases = np.zeros((1, new_neuron_count))
#
#
#     new_links[:net.hidden_end_index, :net.hidden_end_index] = net.links[:net.hidden_end_index, :net.hidden_end_index]
#     new_links[:net.hidden_end_index, -net.output_size:] = net.links[:net.hidden_end_index, -net.output_size:]
#
#     new_weights[:net.hidden_end_index, :net.hidden_end_index] = net.weights[:net.hidden_end_index, :net.hidden_end_index]
#     new_weights[:net.hidden_end_index, -net.output_size:] = net.weights[:net.hidden_end_index, -net.output_size:]
#
#     new_biases[0, :net.hidden_end_index] = net.biases[0, :net.hidden_end_index]
#     new_biases[0, -net.output_size:] = net.biases[0, -net.output_size:]
#
#     new_actFun = []
#     for i in range(net.hidden_end_index):
#         af = net.actFuns[i]
#         if af is None:
#             new_actFun.append(None)
#         else:
#             new_actFun.append(af.copy())
#
#     for i in range(to_add):
#         af = new_actFun[net.input_size + i]
#         new_actFun.append(af.copy())
#
#     for i in range(net.hidden_end_index, net.neuron_count):
#         new_actFun.append(None)
#
#     return ChaosNet(input_size=net.input_size, output_size=net.output_size, links=new_links, weights=new_weights,
#                     biases=new_biases, actFuns=new_actFun, aggrFun=net.aggrFun.copy(), maxit=net.maxit, mutation_radius=net.mutation_radius,
#                     wb_mutation_prob=net.wb_mutation_prob, s_mutation_prob=net.s_mutation_prob, p_mutation_prob=net.p_mutation_prob,
#                     c_prob=net.c_prob, r_prob=net.r_prob)

# def deflate_network(net: ChaosNet):
#     ind_to_preserve = net.get_indices_of_connected_neurons()
#     ind_to_preserve = sorted(ind_to_preserve)
#     ind_to_preserve = np.array(ind_to_preserve).reshape(1, -1)
#
#     new_links = net.links[ind_to_preserve[0, :, None], ind_to_preserve]
#     new_weights = net.weights[ind_to_preserve[0, :, None], ind_to_preserve]
#     new_biases = net.biases[0, ind_to_preserve]
#     new_af = []
#     for i in range(ind_to_preserve.shape[1]):
#         new_af.append(net.actFuns[ind_to_preserve[0, i]])
#
#     return ChaosNet(input_size=net.input_size, output_size=net.output_size, links=new_links, weights=new_weights,
#                     biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, maxit=net.maxit, mutation_radius=net.mutation_radius,
#                     wb_mutation_prob=net.wb_mutation_prob, s_mutation_prob=net.s_mutation_prob, p_mutation_prob=net.p_mutation_prob,
#                     c_prob=net.c_prob, r_prob=net.r_prob)


def gaussian_shift(matrix: np.ndarray, mask: np.ndarray, prob: float, radius: float) -> np.ndarray:
    result = matrix.copy()

    probs = np.random.random(matrix.shape)
    to_change = np.where(probs <= prob)
    change = np.zeros(matrix.shape)
    change[to_change] = 1
    shift = np.random.normal(0, radius, matrix.shape)
    shift = np.multiply(change, shift)
    result += shift
    result = np.multiply(result, mask)

    return result


def uniform_shift(matrix: np.ndarray, mask: np.ndarray, prob: float, minS: float, maxS: float) -> np.ndarray:
    result = matrix.copy()

    probs = np.random.random(matrix.shape)
    to_change = np.where(probs <= prob)
    change = np.zeros(matrix.shape)
    change[to_change] = 1
    shift = np.random.uniform(minS, maxS, matrix.shape)
    shift = np.multiply(change, shift)
    result += shift
    result = np.multiply(result, mask)

    return result

def reroll_matrix(matrix: np.ndarray, mask: np.ndarray, prob: float, minV: float, maxV: float):
    result = matrix.copy()

    probs = np.random.random(matrix.shape)
    to_change = np.where(probs <= prob)
    result[to_change] = np.random.uniform(minV, maxV, matrix.shape)[to_change]
    result = np.multiply(result, mask)

    return result

def reroll_value(p: float, value: float, minV: float, maxV: float):
    result = value

    if random.random() <= p:
        result = random.uniform(minV, maxV)

    return result

def conditional_try_choose_different(p: float, current, options):
    result = current

    if random.random() <= p:
        result = try_choose_different(current, options)

    return result

def conditional_value_swap(prob: float, val1, val2):
    res1 = val1
    res2 = val2

    r = random.random()
    if r <= prob:
        tmp = res1
        res1 = res2
        res2 = tmp

    return res1, res2

# TODO - A - test
def conditional_uniform_value_shift(p: float, value: float, minV: float, maxV: float, frac: float):
    if random.random() <= p:
        spectrum = maxV - minV
        radius = spectrum * frac
        minR = max(minV, value - radius)
        maxR = min(maxV, value + radius)
        value = random.uniform(minR, maxR)

    return value

# TODO - A - test
def add_remove_weights(s_pm: float, weights: np.ndarray, links: np.ndarray, mask):
    probs = np.random.random(links.shape)
    to_change = np.where(probs <= s_pm)
    new_links = links.copy()
    new_links[to_change] = 1 - new_links[to_change]
    new_links = np.multiply(new_links, mask)

    diffs = links - new_links
    added_edges = np.where(diffs == -1)
    #TODO - B - not correct extraction of min/max weights
    minW = np.min(weights)
    maxW = np.max(weights)
    weights[added_edges] = np.random.uniform(minW, maxW, weights.shape)[added_edges]

    links = new_links

    weights = np.multiply(weights, links)

    return weights, links

# TODO - C - to delete
def get_min_max_values_of_matrix_with_mask(matrix: np.ndarray, mask: np.ndarray):
    only_present = matrix[np.where(mask == 1)]
    minW = 0
    maxW = 0
    if only_present.shape[0] != 0:
        minW = np.min(only_present)
        maxW = np.max(only_present)

    return minW, maxW





