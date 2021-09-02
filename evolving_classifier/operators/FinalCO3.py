import random
from math import ceil

from ann_point import HyperparameterRange
from neural_network.ChaosNet import ChaosNet
import numpy as np

from utility.CNDataPoint import CNDataPoint
from utility.Mut_Utility import conditional_value_swap, get_weight_mask
from utility.Utility import choose_without_repetition


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        pass

# TODO - B - remove needless code from here
# TODO - A - jeszcze przejrzyj to
class FinalCO3(CrossoverOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__()
        self.hrange = hrange.copy()

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        possible_cuts = find_possible_cuts99(pointA, pointB, self.hrange)

        cuts = choose_without_repetition(options=possible_cuts, count=2)

        new_A_links, new_A_weights, new_A_biases, new_A_func = get_link_weights_biases_acts7(pointA=pointA, pointB=pointB, cut=cuts[0])
        new_B_links, new_B_weights, new_B_biases, new_B_func = get_link_weights_biases_acts7(pointA=pointA, pointB=pointB, cut=cuts[1])

        swap_pm = 10 ** pointA.swap_prob

        new_A_aggr, new_B_aggr = conditional_value_swap(swap_pm, pointA.aggrFun, pointB.aggrFun)

        # maxIt swap

        new_A_net_it, new_B_net_it = conditional_value_swap(swap_pm, pointA.net_it, pointB.net_it)

        # mutation radius swap

        new_A_mut_rad, new_B_mut_rad = conditional_value_swap(swap_pm, pointA.mutation_radius, pointB.mutation_radius)

        # wb prob swap

        new_A_swap_prob, new_B_swap_prob = conditional_value_swap(swap_pm, pointA.swap_prob, pointB.swap_prob)

        # s prob swap

        new_A_multi_prob, new_B_multi_prob = conditional_value_swap(swap_pm, pointA.multi, pointB.multi)

        # p prob swap

        new_A_p_prob, new_B_p_prob = conditional_value_swap(swap_pm, pointA.p_prob, pointB.p_prob)

        # c prob swap

        new_A_c_prob, new_B_c_prob = conditional_value_swap(swap_pm, pointA.c_prob, pointB.c_prob)

        # r prob swap

        new_A_p_rad, new_B_p_rad = conditional_value_swap(swap_pm, pointA.p_rad, pointB.p_rad)

        # act fun prob

        # new_A_act_prob, new_B_act_prob = conditional_value_swap(pointA.p_mutation_prob, pointA.act_mut_prob, pointB.act_mut_prob)

        pointA = ChaosNet(input_size=pointA.input_size, output_size=pointA.output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_biases, actFuns=new_A_func, aggrFun=new_A_aggr, net_it=new_A_net_it, mutation_radius=new_A_mut_rad,
                          swap_prob=new_A_swap_prob, multi=new_A_multi_prob, p_prob=new_A_p_prob,
                          c_prob=new_A_c_prob, p_rad=new_A_p_rad)

        pointB = ChaosNet(input_size=pointB.input_size, output_size=pointB.output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_biases, actFuns=new_B_func, aggrFun=new_B_aggr, net_it=new_B_net_it, mutation_radius=new_B_mut_rad,
                          swap_prob=new_B_swap_prob, multi=new_B_multi_prob, p_prob=new_B_p_prob,
                          c_prob=new_B_c_prob, p_rad=new_B_p_rad)

        return pointA, pointB


def find_possible_cuts99(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
    possible_cuts = []
    maxh = hrange.max_hidden
    minh = hrange.min_hidden

    for i in range(pointA.hidden_start_index, pointA.hidden_end_index + 1):
        for j in range(pointB.hidden_start_index, pointB.hidden_end_index + 1):
            A_lhc = i - pointA.input_size
            A_rhc = pointA.hidden_count - A_lhc
            B_rhc = pointB.hidden_end_index - j
            B_lhc = pointB.hidden_count - B_rhc

            nrhc = A_lhc + B_rhc

            tol = 2
            if minh <= nrhc <= maxh:
                if min(B_rhc - A_rhc, A_lhc - B_lhc) <= tol:
                    if abs(pointA.hidden_count - (nrhc)) <= tol or abs(pointB.hidden_count - (nrhc)) <= tol:
                        possible_cuts.append([i, A_lhc, j, B_rhc])

    while len(possible_cuts) <= 2:
        possible_cuts.append([pointA.hidden_start_index, 0, pointB.hidden_end_index, 0])

    return possible_cuts

def cut_into_puzzles99(matrix: np.ndarray, i: int, o: int, ync: int, onc: int, left: bool) -> [np.ndarray]:
    currnc = matrix.shape[0]

    if left:
        tt = min(currnc - o, i+ync+onc)
        P1 = matrix[i:tt, i:i+ync]
        P2 = matrix[:i, i:i+ync]
        P3 = matrix[i:i+ync, -o:]
        P4 = matrix[i:i+ync, i+ync:tt]
    else:
        tt = max(i, currnc - o - ync - onc)
        P1 = matrix[tt:-o, -(o+ync):-o]
        P2 = matrix[:i, -(o+ync):-o]
        P3 = matrix[-(o+ync):-o, -o:]
        P4 = matrix[-(o + ync):-o, tt:-(o + ync)]

    return P1, P2, P3, P4


def piece_together_from_puzzles7(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
    left_nc = left_puzzles[0].shape[1]
    right_nc = right_puzzles[0].shape[1]

    h = left_nc + right_nc
    n = i + o + h

    aEnd = i + left_nc

    result = np.zeros((n, n))

    # Put in piece #4
    result[i:i+left_nc, i+left_nc:i+left_nc+left_puzzles[3].shape[1]] = left_puzzles[3]
    result[-(o+right_nc):-o, -(o+right_nc+right_puzzles[3].shape[1]):-(o+right_nc)] = right_puzzles[3]

    # Put in piece #1
    result[i:i+left_puzzles[0].shape[0], i:i+left_nc] = left_puzzles[0]
    result[-(o + right_puzzles[0].shape[0]):-o, aEnd:aEnd+right_nc] = right_puzzles[0]

    # Put in piece #2
    result[:i, i:i+left_nc] = left_puzzles[1]
    result[:i, aEnd:aEnd+right_nc] = right_puzzles[1]

    # Put in piece #3
    result[i:i+left_nc, -o:] = left_puzzles[2]
    result[aEnd:aEnd+right_nc, -o:] = right_puzzles[2]

    return result

def get_link_weights_biases_acts7(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
    input_size = pointA.input_size
    output_size = pointA.output_size

    links = piece_together_from_puzzles7(i=input_size, o=output_size,
                                         left_puzzles=cut_into_puzzles99(matrix=pointA.links, i=input_size, o=output_size,
                                                                         ync=cut[1], onc=cut[3], left=True),
                                         right_puzzles=cut_into_puzzles99(matrix=pointB.links, i=input_size, o=output_size,
                                                                          ync=cut[3], onc=cut[1], left=False))
    links = np.multiply(links, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))

    weights = piece_together_from_puzzles7(i=input_size, o=output_size,
                                           left_puzzles=cut_into_puzzles99(matrix=pointA.weights, i=input_size, o=output_size,
                                                                           ync=cut[1], onc=cut[3], left=True),
                                           right_puzzles=cut_into_puzzles99(matrix=pointB.weights, i=input_size, o=output_size,
                                                                            ync=cut[3], onc=cut[1], left=False))
    weights = np.multiply(weights, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))
    nc = input_size + output_size + cut[1] + cut[3]

    biases = np.zeros((1, nc))
    acts = nc * [None]
    for i in range(cut[1]):
        acts[input_size + i] = pointA.actFuns[input_size + i].copy()
    for i in range(cut[3]):
        acts[input_size + cut[1] + i] = pointB.actFuns[cut[2] + i].copy()

    biases[0, input_size:input_size+cut[1]] = pointA.biases[0, input_size:cut[0]]
    biases[0, input_size+cut[1]:input_size+cut[1]+cut[3]] = pointB.biases[0, cut[2]:-output_size]

    # TODO - C - logika losowania jest odwrócona
    for i in range(output_size):
        swap = random.random()
        if swap <= 0.5:
            biases[0, -output_size + i] = pointA.biases[0, -output_size + i]
        else:
            biases[0, -output_size + i] = pointB.biases[0, -output_size + i]

    return links, weights, biases, acts
