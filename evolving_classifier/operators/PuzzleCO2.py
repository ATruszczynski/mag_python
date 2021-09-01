import random

from ann_point import HyperparameterRange
from evolving_classifier.operators.FinalCO1 import find_possible_cuts4
from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint
from utility.Mut_Utility import conditional_value_swap
from utility.Utility import choose_without_repetition
from utility.Utility2 import *
import numpy as np


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        pass

#TODO - B - AL zamiast A1 etc?
# TODO - B - remove needless code from here
# TODO - B - test

class PuzzleCO2(CrossoverOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__()
        self.hrange = hrange

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:

        possible_cuts = find_possible_cuts4(pointA, pointB, self.hrange)

        # cut1 = possible_cuts[random.randint(0, len(possible_cuts) - 1)]
        # cut2 = possible_cuts[random.randint(0, len(possible_cuts) - 1)]
        cuts = choose_without_repetition(options=possible_cuts, count=2)

        if len(cuts) == 1: # obie sieci mają zero neuronów ukrytych
            cuts.append([1, 0, 1, 0])

        input_size = pointA.input_size
        output_size = pointA.output_size


        C_links, C_weights, C_biases, C_acts = get_link_weights_biases_acts(pointA=pointA, pointB=pointB, cut=cuts[0])
        D_links, D_weights, D_biases, D_acts = get_link_weights_biases_acts(pointA=pointA, pointB=pointB, cut=cuts[1])

        # aggr swap
        C_aggr, D_aggr = conditional_value_swap(0.5, pointA.aggrFun, pointB.aggrFun)

        # maxIt swap

        C_maxit, D_maxit = conditional_value_swap(0.5, pointA.net_it, pointB.net_it)

        # mutation radius swap

        C_mut_rad, D_mut_rad = conditional_value_swap(0.5, pointA.mutation_radius, pointB.mutation_radius)

        # wb prob swap

        C_wb_prob, D_wb_prob = conditional_value_swap(0.5, pointA.depr, pointB.depr)

        # s prob swap

        C_s_prob, D_s_prob = conditional_value_swap(0.5, pointA.multi, pointB.multi)

        # p prob swap

        C_p_prob, D_p_prob = conditional_value_swap(0.5, pointA.p_prob, pointB.p_prob)

        # c prob swap

        C_c_prob, D_c_prob = conditional_value_swap(0.5, pointA.c_prob, pointB.c_prob)

        # r prob swap

        C_r_prob, D_r_prob = conditional_value_swap(0.5, pointA.p_rad, pointB.p_rad)

        # act fun prob

        new_A_act_prob, new_B_act_prob = conditional_value_swap(0.5, pointA.depr_2, pointB.depr_2)


        pointC = ChaosNet(input_size=input_size, output_size=output_size, links=C_links, weights=C_weights,
                          biases=C_biases, actFuns=C_acts, aggrFun=C_aggr, net_it=C_maxit, mutation_radius=C_mut_rad,
                          depr=C_wb_prob, multi=C_s_prob, p_prob=C_p_prob,
                          c_prob=C_c_prob, p_rad=C_r_prob)

        pointD = ChaosNet(input_size=input_size, output_size=output_size, links=D_links, weights=D_weights,
                          biases=D_biases, actFuns=D_acts, aggrFun=D_aggr, net_it=D_maxit, mutation_radius=D_mut_rad,
                          depr=D_wb_prob, multi=D_s_prob, p_prob=D_p_prob,
                          c_prob=D_c_prob, p_rad=D_r_prob)


        return pointC.copy(), pointD.copy()

def get_link_weights_biases_acts(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
    input_size = pointA.input_size
    output_size = pointA.output_size

    # if cut[1] != 0:
    #     lA = cut[0] - pointA.hidden_start_index
    #     rA = pointA.hidden_end_index - cut[0] - cut[1]
    # else:
    #     lA = 0
    #     rA = 0
    #
    # if cut[3] != 0:
    #     lB = cut[2] - pointB.hidden_start_index
    #     rB = pointB.hidden_end_index - cut[2] - cut[3]
    # else:
    #     lB = 0
    #     rB = 0

    # if lA + rB > rA + lB:
    #     tmp = pointA
    #     pointA = pointB
    #     pointB = tmp
    #
    #     tmp = cut[0]
    #     cut[0] = cut[2]
    #     cut[2] = tmp
    #
    #     tmp = cut[1]
    #     cut[1] = cut[3]
    #     cut[3] = tmp


    links = piece_together_from_puzzles2(i=input_size, o=output_size,
                                         left_puzzles=cut_into_puzzles2(matrix=pointA.links, i=input_size, o=output_size,
                                                                      start=cut[0], num=cut[1], lim=cut[3], left=True),
                                         right_puzzles=cut_into_puzzles2(matrix=pointB.links, i=input_size, o=output_size,
                                                                       start=cut[2], num=cut[3], lim=cut[1], left=False))

    weights = piece_together_from_puzzles2(i=input_size, o=output_size,
                                           left_puzzles=cut_into_puzzles2(matrix=pointA.weights, i=input_size, o=output_size,
                                                                        start=cut[0], num=cut[1], lim=cut[3], left=True),
                                           right_puzzles=cut_into_puzzles2(matrix=pointB.weights, i=input_size, o=output_size,
                                                                         start=cut[2], num=cut[3], lim=cut[1], left=False))

    nc = input_size + output_size + cut[1] + cut[3]

    biases = np.zeros((1, nc))
    acts = nc * [None]
    for i in range(cut[1]):
        biases[0, input_size + i] = pointA.biases[0, cut[0] + i].copy()
        acts[input_size + i] = pointA.actFuns[cut[0] + i]
    for i in range(cut[3]):
        biases[0, input_size + cut[1] + i] = pointB.biases[0, cut[2] + i]
        acts[input_size + cut[1] + i] = pointB.actFuns[cut[2] + i].copy()

    for i in range(output_size):
        swap = random.random()
        if swap <= 0.5:
            biases[0, -output_size + i] = pointA.biases[0, -output_size + i]
        else:
            biases[0, -output_size + i] = pointB.biases[0, -output_size + i]

    return links, weights, biases, acts

def piece_together_from_puzzles2(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
    left_nc = left_puzzles[6].shape[0]
    right_nc = right_puzzles[6].shape[0]

    n = i + o + left_nc + right_nc

    aEnd = i + left_nc

    result = np.zeros((n, n))

    # Put in pieces #1
    a1w = left_puzzles[0].shape[1]
    b1w = right_puzzles[0].shape[1]

    result[i:aEnd, -(o + a1w):-o] = left_puzzles[0]
    result[aEnd:-o, i:i+b1w] = right_puzzles[0]

    # Put in pieces #2
    a2h = left_puzzles[1].shape[0]
    b2h = right_puzzles[1].shape[0]

    result[-(o + a2h):-o, i:aEnd] = left_puzzles[1]
    result[i:i+b2h, aEnd:-o] = right_puzzles[1]

    # Put in pieces #3
    a3w = left_puzzles[2].shape[1]
    b3w = right_puzzles[2].shape[1]

    result[i:aEnd, aEnd:aEnd+a3w] = left_puzzles[2]
    result[aEnd:-o, i:i+b3w] = right_puzzles[2]

    # Put in pieces #4
    a4h = left_puzzles[3].shape[0]
    b4h = right_puzzles[3].shape[0]

    result[aEnd:aEnd+a4h, i:aEnd] = left_puzzles[3]
    result[aEnd-b4h:aEnd, aEnd:-o] = right_puzzles[3]

    # Put in pieces #5
    result[i:aEnd, -o:] = left_puzzles[4]
    result[aEnd:-o, -o:] = right_puzzles[4]

    # Put in pieces #6
    result[:i, i:aEnd] = left_puzzles[5]
    result[:i, aEnd:-o] = right_puzzles[5]

    # Put in pieces #7
    result[i:aEnd, i:aEnd] = left_puzzles[6]
    result[aEnd:-o, aEnd:-o] = right_puzzles[6]

    # # Put in pieces #1
    # result[i:i+left_nc, i:i+left_nc] = left_puzzles[0]
    # result[i+left_nc:i+left_nc+right_nc, i+left_nc:i+left_nc+right_nc] = right_puzzles[0]
    #
    # # Put in pieces #2
    # result[:i, i:i+left_nc] = left_puzzles[1]
    # result[:i, i+left_nc:i+left_nc+right_nc] = right_puzzles[1]
    #
    # # Put in pieces #3
    # result[i:i+left_nc, -o:] = left_puzzles[2]
    # result[i+left_nc:i+left_nc+right_nc, -o:] = right_puzzles[2]
    #
    # # Put in pieces #4
    # l4w = left_puzzles[3].shape[1]
    # result[i:i+left_nc, i+left_nc:i+left_nc+l4w] = left_puzzles[3]
    # r4w = right_puzzles[3].shape[1]
    # result[i+left_nc:i+left_nc+right_nc, i+left_nc-r4w:i+left_nc] = right_puzzles[3]
    #
    # # Put in pieces #5
    # l5h = left_puzzles[4].shape[0]
    # result[i+left_nc:i+left_nc+l5h, i:i+left_nc] = left_puzzles[4]
    # r5h = right_puzzles[4].shape[0]
    # result[i+left_nc-r5h:i+left_nc, i+left_nc:i+left_nc+right_nc] = right_puzzles[4]

    return result

def cut_into_puzzles2(matrix: np.ndarray, i: int, o: int, start: int, num: int, lim: int, left: bool) -> [np.ndarray]:
    n = matrix.shape[0]

    end = start + num

    p12si = max(i, start-lim)
    P1 = matrix[start:end, p12si:start]
    P2 = matrix[p12si:start, start:end]

    p34ei = min(n - o, end + lim)
    P3 = matrix[start:end, end:p34ei]
    P4 = matrix[end:p34ei, start:end]

    if not left:
        tmp = P1
        P1 = P3
        P3 = tmp

        tmp = P2
        P2 = P4
        P4 = tmp
    # else:
    #     p12ei = min(n - o, end + lim)
    #     P1 = matrix[start:end, end:p12ei]
    #     P2 = matrix[end:p12ei, start:end]
    #
    #     p34si = max(i, start-lim)
    #     P3 = matrix[start:end, p34si:start]
    #     P4 = matrix[p34si:start, start:end]

    P5 = matrix[start:end, -o:]
    P6 = matrix[:i, start:end]
    P7 = matrix[start:end, start:end]

    return P1, P2, P3, P4, P5, P6, P7