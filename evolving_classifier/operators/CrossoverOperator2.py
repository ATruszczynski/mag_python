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

class FinalCrossoverOperator2(CrossoverOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__()
        self.hrange = hrange

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        possible_cuts = find_possible_cuts4(pointA, pointB, self.hrange)

        cuts = choose_without_repetition(options=possible_cuts, count=2)

        # if len(cuts) == 1: # obie sieci mają zero neuronów ukrytych
        #     cuts.append([1, 0, 1, 0])

        input_size = pointA.input_size
        output_size = pointA.output_size

        new_A_links, new_A_weights, new_A_biases, new_A_func = get_link_weights_biases_acts3(pointA=pointA, pointB=pointB, cut=cuts[0])
        new_B_links, new_B_weights, new_B_biases, new_B_func = get_link_weights_biases_acts3(pointA=pointA, pointB=pointB, cut=cuts[1])

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
def find_possible_cuts5(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
    possible_cuts = []
    maxh = hrange.max_hidden
    minh = hrange.min_hidden
    for sA in range(pointA.hidden_start_index, pointA.hidden_end_index):
        for eA in range(sA, pointA.hidden_end_index + 1):
            for sB in range(pointB.hidden_start_index, pointB.hidden_end_index):
                for eB in range(sB, pointB.hidden_end_index + 1):
                    hL = eA - sA
                    hR = eB - sB

                    if hL == 0 or hR == 0: #TODO - A - usunąć inne wycinanki?
                        continue

                    if hL + hR >= minh and hL + hR <= maxh:
                        possible_cuts.append([sA, hL, sB, hR])

    #TODO - S - tu mogą być ignorowane ograniczenia

    for sA in range(pointA.hidden_start_index, pointA.hidden_end_index):
        for eA in range(sA, pointA.hidden_end_index + 1):
            hL = eA - sA
            if hL > 0 and hL >= minh and hL <= maxh:
                possible_cuts.append([sA, hL, 1, 0])


    for sB in range(pointB.hidden_start_index, pointB.hidden_end_index):
        for eB in range(sB, pointB.hidden_end_index + 1):
            hR = eB - sB
            if hR > 0 and hR >= minh and hR <= maxh:
                possible_cuts.append([1, 0, sB, hR])

    # TODO - S - tu mogą być igrnorowane ograniczenia (a w głównym kodzie trzeba zmienić co się dizeje kiedy jest za mało cieć do wyboru
    possible_cuts.append([1, 0, 1, 0])
    return possible_cuts

def cut_into_puzzles3(matrix: np.ndarray, o: int, start: int, num: int) -> [np.ndarray]:
    P1 = matrix[:, start:start + num]
    P2 = matrix[start:start + num, -o:]

    return P1, P2

def piece_together_from_puzzles3(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
    left_nc = left_puzzles[0].shape[1]
    right_nc = right_puzzles[0].shape[1]

    n = i + o + left_nc + right_nc
    hei = i + left_nc + right_nc
    aEnd = i + left_nc

    result = np.zeros((n, n))

    # Put in pieces #1
    l1h = min(left_puzzles[0].shape[0], hei)
    r1h = min(right_puzzles[0].shape[0], hei)

    result[:l1h, i:aEnd] = left_puzzles[0][:l1h, :]
    result[:r1h, aEnd:-o] = right_puzzles[0][:r1h, :]

    # Put in pieces #2
    result[i:aEnd, -o:] = left_puzzles[1]
    result[aEnd:-o, -o:] = right_puzzles[1]

    return result

def get_link_weights_biases_acts3(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
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


    links = piece_together_from_puzzles3(i=input_size, o=output_size,
                                        left_puzzles=cut_into_puzzles3(matrix=pointA.links, o=output_size,
                                                                      start=cut[0], num=cut[1]),
                                        right_puzzles=cut_into_puzzles3(matrix=pointB.links, o=output_size,
                                                                       start=cut[2], num=cut[3]))
    links = np.multiply(links, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))

    weights = piece_together_from_puzzles3(i=input_size, o=output_size,
                                          left_puzzles=cut_into_puzzles3(matrix=pointA.weights, o=output_size,
                                                                        start=cut[0], num=cut[1]),
                                          right_puzzles=cut_into_puzzles3(matrix=pointB.weights, o=output_size,
                                                                         start=cut[2], num=cut[3]))
    weights = np.multiply(weights, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))
    nc = input_size + output_size + cut[1] + cut[3]

    biases = np.zeros((1, nc))
    acts = nc * [None]
    for i in range(cut[1]):
        biases[0, input_size + i] = pointA.biases[0, cut[0] + i].copy()
        acts[input_size + i] = pointA.actFuns[cut[0] + i]
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
