import random
# from neural_network.ChaosNet import ChaosNet

from utility.Utility import *

# TODO - B - remove needless functions

def change_neuron_count(net: LsmNetwork, hrange: HyperparameterRange, demanded_hidden: int):
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

def increase_neuron_count(net: LsmNetwork, hrange: HyperparameterRange, to_add: int):
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
    for i in range(to_add):
        new_af.append(hrange.actFunSet[random.randint(0, len(hrange.actFunSet) - 1)].copy())

    new_af.extend(net.actFuns[net.hidden_end_index:])

    return LsmNetwork(input_size=input_size, output_size=output_size, links=new_links, weights=new_weights,
                      biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, net_it=net.net_it, mutation_radius=net.mutation_radius,
                      swap_prob=net.swap_prob, multi=net.multi, p_prob=net.p_prob,
                      c_prob=net.c_prob, p_rad=net.p_rad)


def decrease_neuron_count(net: LsmNetwork, to_remove: int):
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

    return LsmNetwork(input_size=net.input_size, output_size=net.output_size, links=new_links, weights=new_weights,
                      biases=new_biases, actFuns=new_af, aggrFun=net.aggrFun, net_it=net.net_it, mutation_radius=net.mutation_radius,
                      swap_prob=net.swap_prob, multi=net.multi, p_prob=net.p_prob,
                      c_prob=net.c_prob, p_rad=net.p_rad)


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

def zero_matrix(matrix: np.ndarray, mask: np.ndarray, prob: float):
    result = matrix.copy()

    probs = np.random.random(matrix.shape)
    to_change = np.where(probs <= prob)
    result[to_change] = np.zeros(matrix.shape)[to_change]
    result = np.multiply(result, mask)

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

def conditional_uniform_value_shift(p: float, value: float, minV: float, maxV: float, frac: float):
    if random.random() <= p:
        spectrum = maxV - minV
        radius = spectrum * frac
        minR = max(minV, value - radius)
        maxR = min(maxV, value + radius)
        value = random.uniform(minR, maxR)

    return value

def conditional_gaussian_value_shift(p: float, value: float, minV: float, maxV: float, frac: float):
    if random.random() <= p:
        spectrum = maxV - minV
        radius = spectrum * frac
        value = random.gauss(value, radius)
        value = max(minV, value)
        value = min(value, maxV)

    return value

def add_or_remove_edges(s_pm: float, links: np.ndarray, weights: np.ndarray, mask, hrange: HyperparameterRange):
    probs = np.random.random(links.shape)
    to_change = np.where(probs <= s_pm)
    swap_links = links.copy()
    swap_links[to_change] = 1 - swap_links[to_change]
    new_links = swap_links


    new_links = np.multiply(new_links, mask)

    diffs = links - new_links
    added_edges = np.where(diffs == -1)
    minW, maxW = get_min_max_values_of_matrix_with_mask(weights, links)

    if minW == 0 and maxW == 0:
        minW = hrange.min_init_wei
        maxW = hrange.max_init_wei

    weights[added_edges] = np.random.uniform(minW, maxW, weights.shape)[added_edges]

    links = new_links

    weights = np.multiply(weights, links)

    return links, weights

def get_min_max_values_of_matrix_with_mask(matrix: np.ndarray, mask: np.ndarray):
    only_present = matrix[np.where(mask == 1)]
    minW = 0
    maxW = 0
    if only_present.shape[0] != 0:
        minW = np.min(only_present)
        maxW = np.max(only_present)

    return minW, maxW





