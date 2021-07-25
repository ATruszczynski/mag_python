import random

# from ann_point.AnnPoint import AnnPoint
# from ann_point.AnnPoint2 import AnnPoint2, point_from_layer_tuples
from neural_network.ChaosNet import ChaosNet
from utility.Mut_Utility import *
from utility.Utility2 import *
# from utility.Utility import get_Xu_matrix
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








class TestCrossoverOperator(CrossoverOperator):
    def __init__(self):
        super().__init__()

    def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
        cut_ori = random.randint(0, 1)


        if pointA.mutation_radius == 0:
            raise Exception()
        if pointB.mutation_radius == 0:
            raise Exception()

        if cut_ori == 0: #vertical cut
            cut = random.randint(pointA.hidden_start_index + 1, pointA.neuron_count - 1)

            # links
            new_A_links = pointA.links.copy()
            new_A_links[:, cut:] = pointB.links[:, cut:].copy()

            new_B_links = pointB.links.copy()
            new_B_links[:, cut:] = pointA.links[:, cut:].copy()


            # weights
            new_A_weights = pointA.weights.copy()
            new_A_weights[:, cut:] = pointB.weights[:, cut:].copy()

            new_B_weights = pointB.weights.copy()
            new_B_weights[:, cut:] = pointA.weights[:, cut:].copy()

            # biases

            new_A_biases = pointA.biases.copy()
            new_A_biases[0, cut:] = pointB.biases[0, cut:].copy()

            new_B_biases = pointB.biases.copy()
            new_B_biases[0, cut:] = pointA.biases[0, cut:].copy()



            # acts

            # new_A_acts = []
            # new_B_acts = []
            # for i in range(pointA.input_size):
            #     new_A_acts.append(None)
            #     new_B_acts.append(None)
            #
            # for i in range(pointA.hidden_start_index, pointA.neuron_count):
            #     if i < cut:
            #         new_A_acts.append(pointA.actFuns[i].copy())
            #         new_B_acts.append(pointB.actFuns[i].copy())
            #     else:
            #         new_A_acts.append(pointB.actFuns[i].copy())
            #         new_B_acts.append(pointA.actFuns[i].copy())

            pass
        else: #horizontal cut
            cut = random.randint(1, pointA.hidden_end_index - 1)

            # links
            new_A_links = pointA.links.copy()
            new_A_links[cut:, :] = pointB.links[cut:, :].copy()

            new_B_links = pointB.links.copy()
            new_B_links[cut:, :] = pointA.links[cut:, :].copy()


            # weights
            new_A_weights = pointA.weights.copy()
            new_A_weights[cut:, :] = pointB.weights[cut:, :].copy()

            new_B_weights = pointB.weights.copy()
            new_B_weights[cut:, :] = pointA.weights[cut:, :].copy()

            # new_A_acts = pointA.actFuns
            # new_B_acts = pointB.actFuns

            new_A_biases = pointA.biases.copy()
            new_B_biases = pointB.biases.copy()

        if pointA.mutation_radius == 0:
            raise Exception()
        if pointB.mutation_radius == 0:
            raise Exception()

        pointA = ChaosNet(input_size=pointA.input_size, output_size=pointA.output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_biases, actFuns=pointA.actFuns, aggrFun=pointA.aggrFun, maxit=pointA.maxit, mutation_radius=pointA.mutation_radius,
                          wb_mutation_prob=pointA.wb_mutation_prob, s_mutation_prob=pointA.s_mutation_prob, p_mutation_prob=pointA.p_mutation_prob)

        pointB = ChaosNet(input_size=pointB.input_size, output_size=pointB.output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_biases, actFuns=pointB.actFuns, aggrFun=pointB.aggrFun, maxit=pointB.maxit, mutation_radius=pointB.mutation_radius,
                          wb_mutation_prob=pointB.wb_mutation_prob, s_mutation_prob=pointB.s_mutation_prob, p_mutation_prob=pointB.p_mutation_prob)

        return pointA, pointB



#TODO exchanges with some probability!
class SimpleCrossoverOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange
        pass

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
        new_A_bias = np.zeros((1, new_A_count))
        new_B_bias = np.zeros((1, new_B_count))

        new_A_bias[0, :cut_A] = pointA.biases[0, :cut_A]
        new_A_bias[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]

        new_B_bias[0, :cut_B] = pointB.biases[0, :cut_B]
        new_B_bias[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]

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

        # aggrFun swap

        new_A_aggr = pointA.aggrFun
        new_B_aggr = pointB.aggrFun
        if random.random() <= 0.5:
            new_A_aggr = pointB.aggrFun
            new_B_aggr = pointA.aggrFun

        # maxIt swap

        new_A_maxit = pointA.maxit
        new_B_maxit = pointB.maxit
        if random.random() <= 0.5:
            new_A_maxit = pointB.maxit
            new_B_maxit = pointA.maxit

        new_A_mut_rad = pointA.mutation_radius
        new_B_mut_rad = pointB.mutation_radius
        if random.random() <= 0.5:
            new_A_mut_rad = pointB.mutation_radius
            new_B_mut_rad = pointA.mutation_radius

        new_A_wb_prob = pointA.wb_mutation_prob
        new_B_wb_prob = pointB.wb_mutation_prob
        if random.random() <= 0.5:
            new_A_wb_prob = pointB.wb_mutation_prob
            new_B_wb_prob = pointA.wb_mutation_prob

        new_A_s_prob = pointA.s_mutation_prob
        new_B_s_prob = pointB.s_mutation_prob
        if random.random() <= 0.5:
            new_A_s_prob = pointB.s_mutation_prob
            new_B_s_prob = pointA.s_mutation_prob

        new_A_p_prob = pointA.p_mutation_prob
        new_B_p_prob = pointB.p_mutation_prob
        if random.random() <= 0.5:
            new_A_p_prob = pointB.p_mutation_prob
            new_B_p_prob = pointA.p_mutation_prob

        new_A_c_prob = pointA.c_prob
        new_B_c_prob = pointB.c_prob
        if random.random() <= 0.5:
            new_A_c_prob = pointB.c_prob
            new_B_c_prob = pointA.c_prob




        pointA = ChaosNet(input_size=input_size, output_size=output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_bias, actFuns=new_A_func, aggrFun=new_A_aggr, maxit=new_A_maxit,
                          mutation_radius=new_A_mut_rad, wb_mutation_prob=new_A_wb_prob, s_mutation_prob=new_A_s_prob,
                          p_mutation_prob=new_A_p_prob, c_prob=new_A_c_prob, r_prob=pointA.r_prob)
        pointB = ChaosNet(input_size=input_size, output_size=output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_bias, actFuns=new_B_func, aggrFun=new_B_aggr, maxit=new_B_maxit,
                          mutation_radius=new_B_mut_rad, wb_mutation_prob=new_B_wb_prob, s_mutation_prob=new_B_s_prob,
                          p_mutation_prob=new_B_p_prob, c_prob=new_B_c_prob, r_prob=pointB.r_prob)

        return pointA, pointB


#TODO exchanges with some probability!
class SimpleCrossoverOperatorHorizontal:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange
        pass

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

        new_A_links[:cut_A, :rows_to_copy_A_A] = pointA.links[:cut_A, :rows_to_copy_A_A]
        new_A_links[-(output_size + B_2):, :rows_to_copy_B_A] = pointB.links[-(output_size + B_2):, :rows_to_copy_B_A]

        new_B_links[:cut_B, :rows_to_copy_B_B] = pointB.links[:cut_B, :rows_to_copy_B_B]
        new_B_links[-(output_size + A_2):, :rows_to_copy_A_B] = pointA.links[-(output_size + A_2):, :rows_to_copy_A_B]

        # weight swap
        new_A_weights = np.zeros((new_A_count, new_A_count))
        new_B_weights = np.zeros((new_B_count, new_B_count))

        new_A_weights[:cut_A, :rows_to_copy_A_A] = pointA.weights[:cut_A, :rows_to_copy_A_A]
        new_A_weights[-(output_size + B_2):, :rows_to_copy_B_A] = pointB.weights[-(output_size + B_2):, :rows_to_copy_B_A]

        new_B_weights[:cut_B, :rows_to_copy_B_B] = pointB.weights[:cut_B, :rows_to_copy_B_B]
        new_B_weights[-(output_size + A_2):, :rows_to_copy_A_B] = pointA.weights[-(output_size + A_2):, :rows_to_copy_A_B]

        # bias swap
        new_A_bias = np.zeros((1, new_A_count))
        new_B_bias = np.zeros((1, new_B_count))

        new_A_bias[0, :cut_A] = pointA.biases[0, :cut_A]
        new_A_bias[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]

        new_B_bias[0, :cut_B] = pointB.biases[0, :cut_B]
        new_B_bias[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]

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

        # aggrFun swap

        new_A_aggr = pointA.aggrFun
        new_B_aggr = pointB.aggrFun
        if random.random() <= 0.5:
            new_A_aggr = pointB.aggrFun
            new_B_aggr = pointA.aggrFun

        # maxIt swap

        new_A_maxit = pointA.maxit
        new_B_maxit = pointB.maxit
        if random.random() <= 0.5:
            new_A_maxit = pointB.maxit
            new_B_maxit = pointA.maxit

        new_A_mut_rad = pointA.mutation_radius
        new_B_mut_rad = pointB.mutation_radius
        if random.random() <= 0.5:
            new_A_mut_rad = pointB.mutation_radius
            new_B_mut_rad = pointA.mutation_radius

        new_A_wb_prob = pointA.wb_mutation_prob
        new_B_wb_prob = pointB.wb_mutation_prob
        if random.random() <= 0.5:
            new_A_wb_prob = pointB.wb_mutation_prob
            new_B_wb_prob = pointA.wb_mutation_prob

        new_A_s_prob = pointA.s_mutation_prob
        new_B_s_prob = pointB.s_mutation_prob
        if random.random() <= 0.5:
            new_A_s_prob = pointB.s_mutation_prob
            new_B_s_prob = pointA.s_mutation_prob

        new_A_p_prob = pointA.p_mutation_prob
        new_B_p_prob = pointB.p_mutation_prob
        if random.random() <= 0.5:
            new_A_p_prob = pointB.p_mutation_prob
            new_B_p_prob = pointA.p_mutation_prob

        new_A_c_prob = pointA.c_prob
        new_B_c_prob = pointB.c_prob
        if random.random() <= 0.5:
            new_A_c_prob = pointB.c_prob
            new_B_c_prob = pointA.c_prob




        pointA = ChaosNet(input_size=input_size, output_size=output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_bias, actFuns=new_A_func, aggrFun=new_A_aggr, maxit=new_A_maxit,
                          mutation_radius=new_A_mut_rad, wb_mutation_prob=new_A_wb_prob, s_mutation_prob=new_A_s_prob,
                          p_mutation_prob=new_A_p_prob, c_prob=new_A_c_prob)
        pointB = ChaosNet(input_size=input_size, output_size=output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_bias, actFuns=new_B_func, aggrFun=new_B_aggr, maxit=new_B_maxit,
                          mutation_radius=new_B_mut_rad, wb_mutation_prob=new_B_wb_prob, s_mutation_prob=new_B_s_prob,
                          p_mutation_prob=new_B_p_prob, c_prob=new_B_c_prob)

        return pointA, pointB


#TODO exchanges with some probability!
class SimpleCrossoverOperator2:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange
        pass

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
        new_A_links = np.multiply(new_A_links, get_weight_mask(input_size=input_size, output_size=output_size, neuron_count=new_A_count))

        new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
        new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]
        new_B_links = np.multiply(new_B_links, get_weight_mask(input_size=input_size, output_size=output_size, neuron_count=new_B_count))

        # weight swap
        new_A_weights = np.zeros((new_A_count, new_A_count))
        new_B_weights = np.zeros((new_B_count, new_B_count))

        new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
        new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]

        new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
        new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]

        # bias swap
        new_A_bias = np.zeros((1, new_A_count))
        new_B_bias = np.zeros((1, new_B_count))

        new_A_bias[0, :cut_A] = pointA.biases[0, :cut_A]
        new_A_bias[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]

        new_B_bias[0, :cut_B] = pointB.biases[0, :cut_B]
        new_B_bias[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]

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

        # aggrFun swap

        new_A_aggr = pointA.aggrFun
        new_B_aggr = pointB.aggrFun
        if random.random() <= 0.5:
            new_A_aggr = pointB.aggrFun
            new_B_aggr = pointA.aggrFun

        # maxIt swap

        new_A_maxit = pointA.maxit
        new_B_maxit = pointB.maxit
        if random.random() <= 0.5:
            new_A_maxit = pointB.maxit
            new_B_maxit = pointA.maxit

        new_A_mut_rad = pointA.mutation_radius
        new_B_mut_rad = pointB.mutation_radius
        if random.random() <= 0.5:
            new_A_mut_rad = pointB.mutation_radius
            new_B_mut_rad = pointA.mutation_radius
        #
        new_A_wb_prob = pointA.wb_mutation_prob
        new_B_wb_prob = pointB.wb_mutation_prob
        if random.random() <= 0.5:
            new_A_wb_prob = pointB.wb_mutation_prob
            new_B_wb_prob = pointA.wb_mutation_prob
        #
        new_A_s_prob = pointA.s_mutation_prob
        new_B_s_prob = pointB.s_mutation_prob
        if random.random() <= 0.5:
            new_A_s_prob = pointB.s_mutation_prob
            new_B_s_prob = pointA.s_mutation_prob
        #
        new_A_p_prob = pointA.p_mutation_prob
        new_B_p_prob = pointB.p_mutation_prob
        if random.random() <= 0.5:
            new_A_p_prob = pointB.p_mutation_prob
            new_B_p_prob = pointA.p_mutation_prob

        pointA = ChaosNet(input_size=input_size, output_size=output_size, links=new_A_links, weights=new_A_weights,
                          biases=new_A_bias, actFuns=new_A_func, aggrFun=new_A_aggr, maxit=new_A_maxit,
                          mutation_radius=new_A_mut_rad, wb_mutation_prob=new_A_wb_prob, s_mutation_prob=new_A_s_prob,
                          p_mutation_prob=new_A_p_prob)
        pointB = ChaosNet(input_size=input_size, output_size=output_size, links=new_B_links, weights=new_B_weights,
                          biases=new_B_bias, actFuns=new_B_func, aggrFun=new_B_aggr, maxit=new_B_maxit,
                          mutation_radius=new_B_mut_rad, wb_mutation_prob=new_B_wb_prob, s_mutation_prob=new_B_s_prob,
                          p_mutation_prob=new_B_p_prob)

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



        # A_1 = i - pointA.hidden_start_index + 1
        # A_2 = pointA.hidden_count - A_1
        #
        # if i < pointA.hidden_end_index and A_1 >= 0 and A_2 >= 0:
        #     for j in range(pointB.hidden_start_index, pointB.hidden_end_index):
        #         B_1 = j - pointB.hidden_start_index + 1
        #         B_2 = pointB.hidden_count - B_1
        #
        #         sum_A = A_1 + B_2
        #         sum_B = A_2 + B_1
        #         # and (i != pointA.hidden_end_index - 1 and j != pointB.hidden_end_index - 1)
        #         if j < pointB.hidden_end_index and B_1 >= 0 and B_2 >= 0 and sum_A >= hrange.min_hidden and sum_B >= hrange.min_hidden and sum_A <= hrange.max_hidden and sum_B <= hrange.max_hidden:
        #             possible_cuts.append((0, i, j, A_1, A_2, B_1, B_2))
        # if i >= pointA.hidden_end_index and i < pointA.neuron_count:
        #     possible_cuts.append((1, i - pointA.neuron_count))

    return possible_cuts

# def conditional_swap(valA, valB):
#

# class SimpleCrossoverOperator:
#     def __init__(self, swap_prob: float = 0.5):
#         self.swap_prob = swap_prob
#         pass
#
#     def crossover(self, pointA: AnnPoint, pointB: AnnPoint) -> [AnnPoint, AnnPoint]:
#         if len(pointA.neuronCounts) <= len(pointB.neuronCounts):
#             pointA = pointA.copy()
#             pointB = pointB.copy()
#         else:
#             tmp = pointA.copy()
#             pointA = pointB.copy()
#             pointB = tmp
#
#         swap_options = []
#         for i in range(1, len(pointA.neuronCounts)):
#             for j in range(i, len(pointB.neuronCounts)):
#                 swap_options.append((i, j))
#
#         swapInds = swap_options[random.randint(0, len(swap_options) - 1)]
#         swapAInd = swapInds[0]
#         swapBInd = swapInds[1]
#
#         layersA = pointA.get_layer_struct()
#         layersB = pointB.get_layer_struct()
#
#         tmp = layersA[swapAInd:]
#         layersA[swapAInd:] = layersB[swapBInd:]
#         layersB[swapBInd:] = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.lossFun.copy()
#             pointA.lossFun = pointB.lossFun.copy()
#             pointB.lossFun = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.learningRate
#             pointA.learningRate = pointB.learningRate
#             pointB.learningRate = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.momCoeff
#             pointA.momCoeff = pointB.momCoeff
#             pointB.momCoeff = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.batchSize
#             pointA.batchSize = pointB.batchSize
#             pointB.batchSize = tmp
#
#         pointA = point_from_layers(layersA, lossFun=pointA.lossFun, learningRate=pointA.learningRate, momCoeff=pointA.momCoeff, batchSize=pointA.batchSize)
#         pointB = point_from_layers(layersB, lossFun=pointB.lossFun, learningRate=pointB.learningRate, momCoeff=pointB.momCoeff, batchSize=pointB.batchSize)
#
#         return pointA, pointB
#
# class LayerSwapCrossoverOperator:
#     def __init__(self, swap_prob: float = 0.5):
#         self.swap_prob = swap_prob
#         pass
#
#     def crossover(self, pointA: AnnPoint, pointB: AnnPoint) -> [AnnPoint, AnnPoint]:
#         layersA = pointA.get_layer_struct()
#         layersB = pointB.get_layer_struct()
#
#         if random.random() < self.swap_prob:
#             tmp = layersA
#             layersA = layersB
#             layersB = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.lossFun.copy()
#             pointA.lossFun = pointB.lossFun.copy()
#             pointB.lossFun = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.learningRate
#             pointA.learningRate = pointB.learningRate
#             pointB.learningRate = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.momCoeff
#             pointA.momCoeff = pointB.momCoeff
#             pointB.momCoeff = tmp
#
#         if random.random() < self.swap_prob:
#             tmp = pointA.batchSize
#             pointA.batchSize = pointB.batchSize
#             pointB.batchSize = tmp
#
#         pointA = point_from_layers(layersA, lossFun=pointA.lossFun, learningRate=pointA.learningRate, momCoeff=pointA.momCoeff, batchSize=pointA.batchSize)
#         pointB = point_from_layers(layersB, lossFun=pointB.lossFun, learningRate=pointB.learningRate, momCoeff=pointB.momCoeff, batchSize=pointB.batchSize)
#
#         return pointA, pointB




