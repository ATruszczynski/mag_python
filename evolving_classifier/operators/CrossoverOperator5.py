import random

from ann_point import HyperparameterRange
from evolving_classifier.operators.CrossoverOperator import find_possible_cuts4
from neural_network.ChaosNet import ChaosNet
import numpy as np

from utility.Mut_Utility import conditional_value_swap, get_weight_mask
from utility.Utility import choose_without_repetition


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        pass

#TODO - B - AL zamiast A1 etc?
#TODO - S - wyrzuć procentowość z selekcji
# TODO - B - remove needless code from here
# TODO - S - test

class FinalCrossoverOperator5(CrossoverOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__()
        self.hrange = hrange

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        possible_cuts = find_possible_cuts7(pointA, pointB, self.hrange)

        cuts = choose_without_repetition(options=possible_cuts, count=2)

        # if len(cuts) == 1: # obie sieci mają zero neuronów ukrytych
        #     cuts.append([1, 0, 1, 0])

        input_size = pointA.input_size
        output_size = pointA.output_size

        new_A_links, new_A_weights, new_A_biases, new_A_func = get_link_weights_biases_acts7(pointA=pointA, pointB=pointB, cut=cuts[0])
        new_B_links, new_B_weights, new_B_biases, new_B_func = get_link_weights_biases_acts7(pointA=pointA, pointB=pointB, cut=cuts[1])

        # cut_A = cut[1]
        # cut_B = cut[2]
        # A_1 = cut[3]
        # A_2 = cut[4]
        # B_1 = cut[5]
        # B_2 = cut[6]
        #
        # new_A_hidden_count = A_1 + B_2
        # new_B_hidden_count = A_2 + B_1
        # new_A_count = pointA.input_size + new_A_hidden_count + pointA.output_size
        # new_B_count = pointB.input_size + new_B_hidden_count + pointB.output_size
        #
        # rows_to_copy_A_A = min(pointA.hidden_end_index, new_A_count - output_size)
        # rows_to_copy_A_B = min(pointA.hidden_end_index, new_B_count - output_size)
        # rows_to_copy_B_A = min(pointB.hidden_end_index, new_A_count - output_size)
        # rows_to_copy_B_B = min(pointB.hidden_end_index, new_B_count - output_size)
        #
        # # link swap
        # new_A_links = np.zeros((new_A_count, new_A_count))
        # new_B_links = np.zeros((new_B_count, new_B_count))
        #
        # new_A_links[:rows_to_copy_A_A, :cut_A] = pointA.links[:rows_to_copy_A_A, :cut_A]
        # new_A_links[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.links[:rows_to_copy_B_A, -(output_size + B_2):]
        #
        # new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
        # new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]
        #
        # # weight swap
        # new_A_weights = np.zeros((new_A_count, new_A_count))
        # new_B_weights = np.zeros((new_B_count, new_B_count))
        #
        # new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
        # new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]
        #
        # new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
        # new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]
        #
        # # bias swap
        # new_A_biases = np.zeros((1, new_A_count))
        # new_B_biases = np.zeros((1, new_B_count))
        #
        # new_A_biases[0, :cut_A] = pointA.biases[0, :cut_A]
        # new_A_biases[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]
        #
        # new_B_biases[0, :cut_B] = pointB.biases[0, :cut_B]
        # new_B_biases[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]
        #
        # # actFun swap
        # new_A_func = input_size * [None]
        # new_B_func = input_size * [None]
        #
        # for i in range(A_1):
        #     new_A_func.append(pointA.actFuns[input_size + i].copy())
        # for i in range(B_2):
        #     new_A_func.append(pointB.actFuns[input_size + B_1 + i].copy())
        # for i in range(output_size):
        #     new_A_func.append(None)
        #
        # for i in range(B_1):
        #     new_B_func.append(pointB.actFuns[input_size + i].copy())
        # for i in range(A_2):
        #     new_B_func.append(pointA.actFuns[input_size + A_1 + i].copy())
        # for i in range(output_size):
        #     new_B_func.append(None)

        new_A_aggr, new_B_aggr = conditional_value_swap(0.5, pointA.aggrFun, pointB.aggrFun)

        # maxIt swap

        new_A_maxit, new_B_maxit = conditional_value_swap(0.5, pointA.maxit, pointB.maxit)

        # mutation radius swap

        new_A_mut_rad, new_B_mut_rad = conditional_value_swap(0.5, pointA.mutation_radius, pointB.mutation_radius)

        # wb prob swap

        new_A_wb_prob, new_B_wb_prob = conditional_value_swap(0.5, pointA.wb_mutation_prob, pointB.wb_mutation_prob)

        # s prob swap

        new_A_s_prob, new_B_s_prob = conditional_value_swap(0.5, pointA.s_mutation_prob, pointB.s_mutation_prob)

        # p prob swap

        new_A_p_prob, new_B_p_prob = conditional_value_swap(0.5, pointA.p_mutation_prob, pointB.p_mutation_prob)

        # c prob swap

        new_A_c_prob, new_B_c_prob = conditional_value_swap(0.5, pointA.c_prob, pointB.c_prob)

        # r prob swap

        new_A_r_prob, new_B_r_prob = conditional_value_swap(0.5, pointA.r_prob, pointB.r_prob)

        pointA = ChaosNet(input_size=pointA.input_size, output_size=pointA.output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_biases, actFuns=new_A_func, aggrFun=new_A_aggr, maxit=new_A_maxit, mutation_radius=new_A_mut_rad,
                          wb_mutation_prob=new_A_wb_prob, s_mutation_prob=new_A_s_prob, p_mutation_prob=new_A_p_prob,
                          c_prob=new_A_c_prob, r_prob=new_A_r_prob)

        pointB = ChaosNet(input_size=pointB.input_size, output_size=pointB.output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_biases, actFuns=new_B_func, aggrFun=new_B_aggr, maxit=new_B_maxit, mutation_radius=new_B_mut_rad,
                          wb_mutation_prob=new_B_wb_prob, s_mutation_prob=new_B_s_prob, p_mutation_prob=new_B_p_prob,
                          c_prob=new_B_c_prob, r_prob=new_B_r_prob)

        return pointA, pointB







#TODO - S - jak reaguje na puste sieci
def find_possible_cuts7(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
    possible_cuts = []
    maxh = hrange.max_hidden
    minh = hrange.min_hidden

    for i in range(pointA.hidden_start_index, pointA.hidden_end_index + 1):
        for j in range(pointB.hidden_start_index, pointB.hidden_end_index + 1):
            lhc = i - pointA.input_size
            rhc = pointB.hidden_end_index - j

            if lhc + rhc >= minh and lhc + rhc <= maxh:
                possible_cuts.append([i, lhc, j, rhc])

    return possible_cuts

def cut_into_puzzles7(matrix: np.ndarray, i:int, o: int, num: int, left: bool) -> [np.ndarray]:
    if left:
        P1 = matrix[i:-o, i:i+num]
        P2 = matrix[:i, i:i+num]
        P3 = matrix[i:i+num, -o:]
    else:
        P1 = matrix[i:-o, -(o + num):-o]
        P2 = matrix[:i, -(o + num):-o]
        P3 = matrix[-(o + num):-o, -o:]

    return P1, P2, P3

def piece_together_from_puzzles7(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
    left_nc = left_puzzles[0].shape[1]
    right_nc = right_puzzles[0].shape[1]

    h = left_nc + right_nc
    n = i + o + h

    aEnd = i + left_nc

    result = np.zeros((n, n))

    # Put in pieces #1
    lc = min(h, left_puzzles[0].shape[0])
    result[i:i+lc, i:i+left_nc] = left_puzzles[0][:lc, :]

    rc = min(h, right_puzzles[0].shape[0])
    result[-(o + rc):-o, aEnd:-o] = right_puzzles[0][-rc:, :]

    # Put in pieces #2
    result[:i, i:aEnd] = left_puzzles[1]
    result[:i, aEnd:-o] = right_puzzles[1]

    # Put in pieces #3
    result[i:i+left_nc, -o:] = left_puzzles[2][:left_nc, :]
    result[-(o + right_nc):-o, -o:] = right_puzzles[2][-right_nc:, :]


    return result

def get_link_weights_biases_acts7(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
    input_size = pointA.input_size
    output_size = pointA.output_size

    if cut[1] != 0:
        lA = cut[0] - pointA.hidden_start_index
        rA = pointA.hidden_end_index - cut[0] - cut[1]
    else:
        lA = 0
        rA = 0

    if cut[3] != 0:
        lB = cut[2] - pointB.hidden_start_index
        rB = pointB.hidden_end_index - cut[2] - cut[3]
    else:
        lB = 0
        rB = 0

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


    links = piece_together_from_puzzles7(i=input_size, o=output_size,
                                        left_puzzles=cut_into_puzzles7(matrix=pointA.links, i=input_size, o=output_size,
                                                                       num=cut[1], left=True),
                                        right_puzzles=cut_into_puzzles7(matrix=pointB.links, i=input_size, o=output_size,
                                                                       num=cut[3], left=False))
    links = np.multiply(links, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))

    weights = piece_together_from_puzzles7(i=input_size, o=output_size,
                                          left_puzzles=cut_into_puzzles7(matrix=pointA.weights, i=input_size, o=output_size,
                                                                        num=cut[1], left=True),
                                          right_puzzles=cut_into_puzzles7(matrix=pointB.weights, i=input_size, o=output_size,
                                                                         num=cut[3], left=False))
    weights = np.multiply(weights, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))
    nc = input_size + output_size + cut[1] + cut[3]

    biases = np.zeros((1, nc))
    acts = nc * [None]
    for i in range(cut[1]):
        biases[0, input_size + i] = pointA.biases[0, input_size + i]
        acts[input_size + i] = pointA.actFuns[input_size + i].copy()
    for i in range(cut[3]):
        biases[0, input_size + cut[1] + i] = pointB.biases[0, cut[2] + i]
        acts[input_size + cut[1] + i] = pointB.actFuns[cut[2] + i].copy()

    # TODO - C - logika losowania jest odwrócona
    for i in range(output_size):
        swap = random.random()
        if swap <= 0.5:
            biases[0, -output_size + i] = pointA.biases[0, -output_size + i]
        else:
            biases[0, -output_size + i] = pointB.biases[0, -output_size + i]

    return links, weights, biases, acts
