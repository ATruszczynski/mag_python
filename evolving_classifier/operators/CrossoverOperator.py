import random
from neural_network.ChaosNet import ChaosNet
from utility.Mut_Utility import *
from utility.Utility2 import *
import numpy as np


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        pass


class FinalCrossoverOperator(CrossoverOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__()
        self.hrange = hrange

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        possible_cuts = find_possible_cuts(pointA, pointB, self.hrange)

        cut = possible_cuts[random.randint(0, len(possible_cuts) - 1)]

        input_size = pointA.input_size
        output_size = pointA.output_size

        cut_A = cut[1]
        cut_B = cut[2]
        A_1 = cut[3]
        A_2 = cut[4]
        B_1 = cut[5]
        B_2 = cut[6]

        new_A_hidden_count = A_1 + B_2
        new_B_hidden_count = A_2 + B_1
        new_A_count = pointA.input_size + new_A_hidden_count + pointA.output_size
        new_B_count = pointB.input_size + new_B_hidden_count + pointB.output_size

        rows_to_copy_A_A = min(pointA.hidden_end_index, new_A_count - output_size)
        rows_to_copy_A_B = min(pointA.hidden_end_index, new_B_count - output_size)
        rows_to_copy_B_A = min(pointB.hidden_end_index, new_A_count - output_size)
        rows_to_copy_B_B = min(pointB.hidden_end_index, new_B_count - output_size)

        # link swap
        new_A_links = np.zeros((new_A_count, new_A_count))
        new_B_links = np.zeros((new_B_count, new_B_count))

        new_A_links[:rows_to_copy_A_A, :cut_A] = pointA.links[:rows_to_copy_A_A, :cut_A]
        new_A_links[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.links[:rows_to_copy_B_A, -(output_size + B_2):]

        new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
        new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]

        # weight swap
        new_A_weights = np.zeros((new_A_count, new_A_count))
        new_B_weights = np.zeros((new_B_count, new_B_count))

        new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
        new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]

        new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
        new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]

        # bias swap
        new_A_biases = np.zeros((1, new_A_count))
        new_B_biases = np.zeros((1, new_B_count))

        new_A_biases[0, :cut_A] = pointA.biases[0, :cut_A]
        new_A_biases[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]

        new_B_biases[0, :cut_B] = pointB.biases[0, :cut_B]
        new_B_biases[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]

        # actFun swap
        new_A_func = input_size * [None]
        new_B_func = input_size * [None]

        for i in range(A_1):
            new_A_func.append(pointA.actFuns[input_size + i].copy())
        for i in range(B_2):
            new_A_func.append(pointB.actFuns[input_size + B_1 + i].copy())
        for i in range(output_size):
            new_A_func.append(None)

        for i in range(B_1):
            new_B_func.append(pointB.actFuns[input_size + i].copy())
        for i in range(A_2):
            new_B_func.append(pointA.actFuns[input_size + A_1 + i].copy())
        for i in range(output_size):
            new_B_func.append(None)

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







def find_possible_cuts(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
    possible_cuts = []
    for i in range(pointA.hidden_start_index, pointA.hidden_end_index + 1):
        A_1 = i - pointA.hidden_start_index
        A_2 = pointA.hidden_count - A_1

        if A_1 >= 0 and A_2 >= 0:
            for j in range(pointB.hidden_start_index, pointB.hidden_end_index + 1):
                if i == pointA.hidden_end_index and j == pointB.hidden_end_index:
                    continue
                B_1 = j - pointB.hidden_start_index
                B_2 = pointB.hidden_count - B_1

                sum_A = A_1 + B_2
                sum_B = A_2 + B_1
                if B_1 >= 0 and B_2 >= 0 and sum_A >= hrange.min_hidden and sum_B >= hrange.min_hidden and sum_A <= hrange.max_hidden and sum_B <= hrange.max_hidden:
                    possible_cuts.append((0, i, j, A_1, A_2, B_1, B_2))

    for i in range(pointA.output_size):
        possible_cuts.append((0, pointA.hidden_end_index + i, pointB.hidden_end_index + i, pointA.hidden_count, 0, pointB.hidden_count, 0))

    return possible_cuts