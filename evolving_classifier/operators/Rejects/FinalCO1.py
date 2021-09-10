# import random
# from neural_network.ChaosNet import ChaosNet
# from utility.Mut_Utility import *
# from utility.Utility2 import *
# import numpy as np
#
#
# class CrossoverOperator:
#     def __init__(self):
#         pass
#
#     def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
#         pass
#

#
# class FinalCO1(CrossoverOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__()
#         self.hrange = hrange
#
#     def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
#         possible_cuts = find_possible_cuts(pointA, pointB, self.hrange)
#
#         cut = possible_cuts[random.randint(0, len(possible_cuts) - 1)]
#
#         input_size = pointA.input_size
#         output_size = pointA.output_size
#
#         cut_A = cut[1]
#         cut_B = cut[2]
#         A_1 = cut[3]
#         A_2 = cut[4]
#         B_1 = cut[5]
#         B_2 = cut[6]
#
#         new_A_hidden_count = A_1 + B_2
#         new_B_hidden_count = A_2 + B_1
#         new_A_count = pointA.input_size + new_A_hidden_count + pointA.output_size
#         new_B_count = pointB.input_size + new_B_hidden_count + pointB.output_size
#
#         rows_to_copy_A_A = min(pointA.hidden_end_index, new_A_count - output_size)
#         rows_to_copy_A_B = min(pointA.hidden_end_index, new_B_count - output_size)
#         rows_to_copy_B_A = min(pointB.hidden_end_index, new_A_count - output_size)
#         rows_to_copy_B_B = min(pointB.hidden_end_index, new_B_count - output_size)
#
#         # link swap
#         new_A_links = np.zeros((new_A_count, new_A_count))
#         new_B_links = np.zeros((new_B_count, new_B_count))
#
#         new_A_links[:rows_to_copy_A_A, :cut_A] = pointA.links[:rows_to_copy_A_A, :cut_A]
#         new_A_links[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.links[:rows_to_copy_B_A, -(output_size + B_2):]
#         new_A_links = np.multiply(new_A_links, get_weight_mask(input_size, output_size, new_A_links.shape[0]))
#
#         new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
#         new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]
#         new_B_links = np.multiply(new_B_links, get_weight_mask(input_size, output_size, new_B_links.shape[0]))
#
#         # weight swap
#         new_A_weights = np.zeros((new_A_count, new_A_count))
#         new_B_weights = np.zeros((new_B_count, new_B_count))
#
#         new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
#         new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]
#         new_A_weights = np.multiply(new_A_weights, get_weight_mask(input_size, output_size, new_A_weights.shape[0]))
#
#         new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
#         new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]
#         new_B_weights = np.multiply(new_B_weights, get_weight_mask(input_size, output_size, new_B_weights.shape[0]))
#
#         # bias swap
#         new_A_biases = np.zeros((1, new_A_count))
#         new_B_biases = np.zeros((1, new_B_count))
#
#         new_A_biases[0, :cut_A] = pointA.biases[0, :cut_A]
#         new_A_biases[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]
#
#         new_B_biases[0, :cut_B] = pointB.biases[0, :cut_B]
#         new_B_biases[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]
#
#         # actFun swap
#         new_A_func = input_size * [None]
#         new_B_func = input_size * [None]
#
#         for i in range(A_1):
#             new_A_func.append(pointA.actFuns[input_size + i].copy())
#         for i in range(B_2):
#             new_A_func.append(pointB.actFuns[input_size + B_1 + i].copy())
#         for i in range(output_size):
#             new_A_func.append(None)
#
#         for i in range(B_1):
#             new_B_func.append(pointB.actFuns[input_size + i].copy())
#         for i in range(A_2):
#             new_B_func.append(pointA.actFuns[input_size + A_1 + i].copy())
#         for i in range(output_size):
#             new_B_func.append(None)
#
#         new_A_aggr, new_B_aggr = conditional_value_swap(0.5, pointA.aggrFun, pointB.aggrFun)
#
#         # maxIt swap
#
#         new_A_maxit, new_B_maxit = conditional_value_swap(0.5, pointA.net_it, pointB.net_it)
#
#         # mutation radius swap
#
#         new_A_mut_rad, new_B_mut_rad = conditional_value_swap(0.5, pointA.mutation_radius, pointB.mutation_radius)
#
#         # wb prob swap
#
#         new_A_wb_prob, new_B_wb_prob = conditional_value_swap(0.5, pointA.swap_prob, pointB.swap_prob)
#
#         # s prob swap
#
#         new_A_s_prob, new_B_s_prob = conditional_value_swap(0.5, pointA.multi, pointB.multi)
#
#         # p prob swap
#
#         new_A_p_prob, new_B_p_prob = conditional_value_swap(0.5, pointA.p_prob, pointB.p_prob)
#
#         # c prob swap
#
#         new_A_c_prob, new_B_c_prob = conditional_value_swap(0.5, pointA.c_prob, pointB.c_prob)
#
#         # r prob swap
#
#         new_A_r_prob, new_B_r_prob = conditional_value_swap(0.5, pointA.p_rad, pointB.p_rad)
#
#         # act fun prob
#
#         # new_A_act_prob, new_B_act_prob = conditional_value_swap(0.5, pointA.act_mut_prob, pointB.act_mut_prob)
#
#         pointA = ChaosNet(input_size=pointA.input_size, output_size=pointA.output_size, links=new_A_links, weights=new_A_weights,
#                           biases=new_A_biases, actFuns=new_A_func, aggrFun=new_A_aggr, net_it=new_A_maxit, mutation_radius=new_A_mut_rad,
#                           swap_prob=new_A_wb_prob, multi=new_A_s_prob, p_prob=new_A_p_prob,
#                           c_prob=new_A_c_prob, p_rad=new_A_r_prob)
#
#         pointB = ChaosNet(input_size=pointB.input_size, output_size=pointB.output_size, links=new_B_links, weights=new_B_weights,
#                           biases=new_B_biases, actFuns=new_B_func, aggrFun=new_B_aggr, net_it=new_B_maxit, mutation_radius=new_B_mut_rad,
#                           swap_prob=new_B_wb_prob, multi=new_B_s_prob, p_prob=new_B_p_prob,
#                           c_prob=new_B_c_prob, p_rad=new_B_r_prob)
#
#         return pointA, pointB
#
#
#
#
#
#
# #TODO - C - pierwszy element wyjścia chyba nie ma już sensu
# def find_possible_cuts(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
#     possible_cuts = []
#     for i in range(pointA.hidden_start_index, pointA.hidden_end_index + 1):
#         A_1 = i - pointA.hidden_start_index
#         A_2 = pointA.hidden_count - A_1
#
#         if A_1 >= 0 and A_2 >= 0:
#             for j in range(pointB.hidden_start_index, pointB.hidden_end_index + 1):
#                 if i == pointA.hidden_end_index and j == pointB.hidden_end_index:
#                     continue
#                 B_1 = j - pointB.hidden_start_index
#                 B_2 = pointB.hidden_count - B_1
#
#                 sum_A = A_1 + B_2
#                 sum_B = A_2 + B_1
#                 if B_1 >= 0 and B_2 >= 0 and sum_A >= hrange.min_hidden and sum_B >= hrange.min_hidden and sum_A <= hrange.max_hidden and sum_B <= hrange.max_hidden:
#                     possible_cuts.append((0, i, j, A_1, A_2, B_1, B_2))
#
#     for i in range(pointA.output_size):
#         possible_cuts.append((0, pointA.hidden_end_index + i, pointB.hidden_end_index + i, pointA.hidden_count, 0, pointB.hidden_count, 0))
#
#     return possible_cuts
# #
# # def find_possible_cuts3(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
# #     possible_cuts = []
# #     for i in range(pointA.hidden_start_index, pointA.hidden_end_index + 1):
# #         A_1 = i - pointA.hidden_start_index
# #         A_2 = pointA.hidden_count - A_1
# #
# #         if A_1 >= 0 and A_2 >= 0: #TODO - C - czy któreś może być ujemne?
# #             for j in range(pointB.hidden_start_index, pointB.hidden_end_index + 1):
# #                 if i == pointA.hidden_end_index and j == pointB.hidden_end_index:
# #                     continue
# #                 B_1 = j - pointB.hidden_start_index
# #                 B_2 = pointB.hidden_count - B_1
# #
# #                 sum_A = A_1 + B_2
# #                 sum_B = A_2 + B_1
# #                 if B_1 >= 0 and B_2 >= 0:
# #                     possible_cuts.append((0, i, j, A_1, A_2, B_1, B_2))
# #
# #     for i in range(pointA.output_size):
# #         possible_cuts.append((0, pointA.hidden_end_index + i, pointB.hidden_end_index + i, pointA.hidden_count, 0, pointB.hidden_count, 0))
# #
# #     return possible_cuts
# #
# #
# #
# #
# #
# #
# #
# # class FinalCrossoverOperator2(CrossoverOperator):
# #     def __init__(self, hrange: HyperparameterRange):
# #         super().__init__()
# #         self.hrange = hrange
# #
# #     def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
# #         if abs(pointA.c_prob - (-0.05795092320281692)) < 1e-6 and pointA.hidden_count == 1 and pointB.hidden_count == 0:
# #             iii = 1
# #         possible_cuts = find_possible_cuts_puzzles(pointA, pointB, self.hrange)
# #
# #         # cut1 = possible_cuts[random.randint(0, len(possible_cuts) - 1)]
# #         # cut2 = possible_cuts[random.randint(0, len(possible_cuts) - 1)]
# #         cuts = choose_without_repetition(options=possible_cuts, count=2)
# #
# #         # if len(cuts) == 1: # obie sieci mają zero neuronów ukrytych
# #         #     cuts.append([1, 0, 1, 0])
# #
# #         input_size = pointA.input_size
# #         output_size = pointA.output_size
# #
# #
# #         C_links, C_weights, C_biases, C_acts = get_link_weights_biases_acts(pointA=pointA, pointB=pointB, cut=cuts[0])
# #         D_links, D_weights, D_biases, D_acts = get_link_weights_biases_acts(pointA=pointA, pointB=pointB, cut=cuts[1])
# #
# #         # aggr swap
# #         C_aggr, D_aggr = conditional_value_swap(0.5, pointA.aggrFun, pointB.aggrFun)
# #
# #         # maxIt swap
# #
# #         C_maxit, D_maxit = conditional_value_swap(0.5, pointA.maxit, pointB.maxit)
# #
# #         # mutation radius swap
# #
# #         C_mut_rad, D_mut_rad = conditional_value_swap(0.5, pointA.mutation_radius, pointB.mutation_radius)
# #
# #         # wb prob swap
# #
# #         C_wb_prob, D_wb_prob = conditional_value_swap(0.5, pointA.wb_mutation_prob, pointB.wb_mutation_prob)
# #
# #         # s prob swap
# #
# #         C_s_prob, D_s_prob = conditional_value_swap(0.5, pointA.s_mutation_prob, pointB.s_mutation_prob)
# #
# #         # p prob swap
# #
# #         C_p_prob, D_p_prob = conditional_value_swap(0.5, pointA.p_mutation_prob, pointB.p_mutation_prob)
# #
# #         # c prob swap
# #
# #         C_c_prob, D_c_prob = conditional_value_swap(0.5, pointA.c_prob, pointB.c_prob)
# #
# #         # r prob swap
# #
# #         C_r_prob, D_r_prob = conditional_value_swap(0.5, pointA.r_prob, pointB.r_prob)
# #
# #
# #         pointC = ChaosNet(input_size=input_size, output_size=output_size, links=C_links, weights=C_weights,
# #                           biases=C_biases, actFuns=C_acts, aggrFun=C_aggr, maxit=C_maxit, mutation_radius=C_mut_rad,
# #                           wb_mutation_prob=C_wb_prob, s_mutation_prob=C_s_prob, p_mutation_prob=C_p_prob,
# #                           c_prob=C_c_prob, r_prob=C_r_prob)
# #
# #         pointD = ChaosNet(input_size=input_size, output_size=output_size, links=D_links, weights=D_weights,
# #                           biases=D_biases, actFuns=D_acts, aggrFun=D_aggr, maxit=D_maxit, mutation_radius=D_mut_rad,
# #                           wb_mutation_prob=D_wb_prob, s_mutation_prob=D_s_prob, p_mutation_prob=D_p_prob,
# #                           c_prob=D_c_prob, r_prob=D_r_prob)
# #
# #         return pointC, pointD
# #
# #
# #         # input_size = pointA.input_size
# #         # output_size = pointA.output_size
# #         #
# #         # cut_A = cut[1]
# #         # cut_B = cut[2]
# #         # A_1 = cut[3]
# #         # A_2 = cut[4]
# #         # B_1 = cut[5]
# #         # B_2 = cut[6]
# #         #
# #         # new_A_hidden_count = A_1 + B_2
# #         # new_B_hidden_count = A_2 + B_1
# #         # new_A_count = pointA.input_size + new_A_hidden_count + pointA.output_size
# #         # new_B_count = pointB.input_size + new_B_hidden_count + pointB.output_size
# #         #
# #         # rows_to_copy_A_A = min(pointA.hidden_end_index, new_A_count - output_size)
# #         # rows_to_copy_A_B = min(pointA.hidden_end_index, new_B_count - output_size)
# #         # rows_to_copy_B_A = min(pointB.hidden_end_index, new_A_count - output_size)
# #         # rows_to_copy_B_B = min(pointB.hidden_end_index, new_B_count - output_size)
# #         #
# #         # # link swap
# #         # new_A_links = np.zeros((new_A_count, new_A_count))
# #         # new_B_links = np.zeros((new_B_count, new_B_count))
# #         #
# #         # new_A_links[:rows_to_copy_A_A, :cut_A] = pointA.links[:rows_to_copy_A_A, :cut_A]
# #         # new_A_links[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.links[:rows_to_copy_B_A, -(output_size + B_2):]
# #         #
# #         # new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
# #         # new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]
# #         #
# #         # # weight swap
# #         # new_A_weights = np.zeros((new_A_count, new_A_count))
# #         # new_B_weights = np.zeros((new_B_count, new_B_count))
# #         #
# #         # new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
# #         # new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]
# #         #
# #         # new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
# #         # new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]
# #         #
# #         # # bias swap
# #         # new_A_biases = np.zeros((1, new_A_count))
# #         # new_B_biases = np.zeros((1, new_B_count))
# #         #
# #         # new_A_biases[0, :cut_A] = pointA.biases[0, :cut_A]
# #         # new_A_biases[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]
# #         #
# #         # new_B_biases[0, :cut_B] = pointB.biases[0, :cut_B]
# #         # new_B_biases[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]
# #         #
# #         # # actFun swap
# #         # new_A_func = input_size * [None]
# #         # new_B_func = input_size * [None]
# #         #
# #         # for i in range(A_1):
# #         #     new_A_func.append(pointA.actFuns[input_size + i].copy())
# #         # for i in range(B_2):
# #         #     new_A_func.append(pointB.actFuns[input_size + B_1 + i].copy())
# #         # for i in range(output_size):
# #         #     new_A_func.append(None)
# #         #
# #         # for i in range(B_1):
# #         #     new_B_func.append(pointB.actFuns[input_size + i].copy())
# #         # for i in range(A_2):
# #         #     new_B_func.append(pointA.actFuns[input_size + A_1 + i].copy())
# #         # for i in range(output_size):
# #         #     new_B_func.append(None)
# #         #
# #         # new_A_aggr, new_B_aggr = conditional_value_swap(0.5, pointA.aggrFun, pointB.aggrFun)
# #         #
# #         # # maxIt swap
# #         #
# #         # new_A_maxit, new_B_maxit = conditional_value_swap(0.5, pointA.maxit, pointB.maxit)
# #         #
# #         # # mutation radius swap
# #         #
# #         # new_A_mut_rad, new_B_mut_rad = conditional_value_swap(0.5, pointA.mutation_radius, pointB.mutation_radius)
# #         #
# #         # # wb prob swap
# #         #
# #         # new_A_wb_prob, new_B_wb_prob = conditional_value_swap(0.5, pointA.wb_mutation_prob, pointB.wb_mutation_prob)
# #         #
# #         # # s prob swap
# #         #
# #         # new_A_s_prob, new_B_s_prob = conditional_value_swap(0.5, pointA.s_mutation_prob, pointB.s_mutation_prob)
# #         #
# #         # # p prob swap
# #         #
# #         # new_A_p_prob, new_B_p_prob = conditional_value_swap(0.5, pointA.p_mutation_prob, pointB.p_mutation_prob)
# #         #
# #         # # c prob swap
# #         #
# #         # new_A_c_prob, new_B_c_prob = conditional_value_swap(0.5, pointA.c_prob, pointB.c_prob)
# #         #
# #         # # r prob swap
# #         #
# #         # new_A_r_prob, new_B_r_prob = conditional_value_swap(0.5, pointA.r_prob, pointB.r_prob)
# #         #
# #         # pointA = ChaosNet(input_size=pointA.input_size, output_size=pointA.output_size, links=new_A_links, weights=new_A_weights,
# #         #                   biases=new_A_biases, actFuns=new_A_func, aggrFun=new_A_aggr, maxit=new_A_maxit, mutation_radius=new_A_mut_rad,
# #         #                   wb_mutation_prob=new_A_wb_prob, s_mutation_prob=new_A_s_prob, p_mutation_prob=new_A_p_prob,
# #         #                   c_prob=new_A_c_prob, r_prob=new_A_r_prob)
# #         #
# #         # pointB = ChaosNet(input_size=pointB.input_size, output_size=pointB.output_size, links=new_B_links, weights=new_B_weights,
# #         #                   biases=new_B_biases, actFuns=new_B_func, aggrFun=new_B_aggr, maxit=new_B_maxit, mutation_radius=new_B_mut_rad,
# #         #                   wb_mutation_prob=new_B_wb_prob, s_mutation_prob=new_B_s_prob, p_mutation_prob=new_B_p_prob,
# #         #                   c_prob=new_B_c_prob, r_prob=new_B_r_prob)
# #         #
# #         # return pointA, pointB
# #
# # def get_link_weights_biases_acts(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
# #     input_size = pointA.input_size
# #     output_size = pointA.output_size
# #
# #     if cut[1] != 0:
# #         lA = cut[0] - pointA.hidden_start_index
# #         rA = pointA.hidden_end_index - cut[0] - cut[1]
# #     else:
# #         lA = 0
# #         rA = 0
# #
# #     if cut[3] != 0:
# #         lB = cut[2] - pointB.hidden_start_index
# #         rB = pointB.hidden_end_index - cut[2] - cut[3]
# #     else:
# #         lB = 0
# #         rB = 0
# #
# #     # if lA + rB > rA + lB:
# #     #     tmp = pointA
# #     #     pointA = pointB
# #     #     pointB = tmp
# #     #
# #     #     tmp = cut[0]
# #     #     cut[0] = cut[2]
# #     #     cut[2] = tmp
# #     #
# #     #     tmp = cut[1]
# #     #     cut[1] = cut[3]
# #     #     cut[3] = tmp
# #
# #
# #     links = piece_together_from_puzzles(i=input_size, o=output_size,
# #                                         left_puzzles=cut_into_puzzles(matrix=pointA.links, i=input_size, o=output_size,
# #                                                                       start=cut[0], num=cut[1], lim=cut[3], left=True),
# #                                         right_puzzles=cut_into_puzzles(matrix=pointB.links, i=input_size, o=output_size,
# #                                                                        start=cut[2], num=cut[3], lim=cut[1], left=False))
# #
# #     weights = piece_together_from_puzzles(i=input_size, o=output_size,
# #                                         left_puzzles=cut_into_puzzles(matrix=pointA.weights, i=input_size, o=output_size,
# #                                                                       start=cut[0], num=cut[1], lim=cut[3], left=True),
# #                                         right_puzzles=cut_into_puzzles(matrix=pointB.weights, i=input_size, o=output_size,
# #                                                                        start=cut[2], num=cut[3], lim=cut[1], left=False))
# #
# #     nc = input_size + output_size + cut[1] + cut[3]
# #
# #     biases = np.zeros((1, nc))
# #     acts = nc * [None]
# #     for i in range(cut[1]):
# #         biases[0, input_size + i] = pointA.biases[0, cut[0] + i].copy()
# #         acts[input_size + i] = pointA.actFuns[cut[0] + i]
# #     for i in range(cut[3]):
# #         biases[0, input_size + cut[1] + i] = pointB.biases[0, cut[2] + i]
# #         acts[input_size + cut[1] + i] = pointB.actFuns[cut[2] + i].copy()
# #
# #     for i in range(output_size):
# #         swap = random.random()
# #         if swap <= 0.5:
# #             biases[0, -output_size + i] = pointA.biases[0, -output_size + i]
# #         else:
# #             biases[0, -output_size + i] = pointB.biases[0, -output_size + i]
# #
# #     return links, weights, biases, acts
# #
# #
# #
# #
# #
# # def piece_together_from_puzzles(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
# #     left_nc = left_puzzles[0].shape[0]
# #     right_nc = right_puzzles[0].shape[0]
# #
# #     n = i + o + left_nc + right_nc
# #
# #     result = np.zeros((n, n))
# #
# #     # Put in pieces #1
# #     result[i:i+left_nc, i:i+left_nc] = left_puzzles[0]
# #     result[i+left_nc:i+left_nc+right_nc, i+left_nc:i+left_nc+right_nc] = right_puzzles[0]
# #
# #     # Put in pieces #2
# #     result[:i, i:i+left_nc] = left_puzzles[1]
# #     result[:i, i+left_nc:i+left_nc+right_nc] = right_puzzles[1]
# #
# #     # Put in pieces #3
# #     result[i:i+left_nc, -o:] = left_puzzles[2]
# #     result[i+left_nc:i+left_nc+right_nc, -o:] = right_puzzles[2]
# #
# #     # Put in pieces #4
# #     l4w = left_puzzles[3].shape[1]
# #     result[i:i+left_nc, i+left_nc:i+left_nc+l4w] = left_puzzles[3]
# #     r4w = right_puzzles[3].shape[1]
# #     result[i+left_nc:i+left_nc+right_nc, i+left_nc-r4w:i+left_nc] = right_puzzles[3]
# #
# #     # Put in pieces #5
# #     l5h = left_puzzles[4].shape[0]
# #     result[i+left_nc:i+left_nc+l5h, i:i+left_nc] = left_puzzles[4]
# #     r5h = right_puzzles[4].shape[0]
# #     result[i+left_nc-r5h:i+left_nc, i+left_nc:i+left_nc+right_nc] = right_puzzles[4]
# #
# #     return result
# #
# #
# def find_possible_cuts4(pointA: ChaosNet, pointB: ChaosNet, hrange: HyperparameterRange):
#     possible_cuts = []
#     maxh = hrange.max_hidden
#     minh = hrange.min_hidden
#     for sA in range(pointA.hidden_start_index, pointA.hidden_end_index):
#         for eA in range(sA, pointA.hidden_end_index + 1):
#             for sB in range(pointB.hidden_start_index, pointB.hidden_end_index):
#                 for eB in range(sB, pointB.hidden_end_index + 1):
#                     hL = eA - sA
#                     hR = eB - sB
#
#                     if hL == 0 or hR == 0:
#                         continue
#
#                     if hL + hR >= minh and hL + hR <= maxh:
#                         possible_cuts.append([sA, hL, sB, hR])
#
#
#     # for sA in range(pointA.hidden_start_index, pointA.hidden_end_index):
#     #     for eA in range(sA, pointA.hidden_end_index + 1):
#     #         hL = eA - sA
#     #         if hL > 0 and hL >= minh and hL <= maxh:
#     #             possible_cuts.append([sA, hL, 0, 0])
#     #
#     #
#     # for sB in range(pointB.hidden_start_index, pointB.hidden_end_index):
#     #     for eB in range(sB, pointB.hidden_end_index + 1):
#     #         hR = eB - sB
#     #         if hR > 0 and hR >= minh and hR <= maxh:
#     #             possible_cuts.append([0, 0, sB, hR])
#
#
#     while len(possible_cuts) < 2:
#         possible_cuts.append([pointA.input_size, pointA.hidden_count, pointB.input_size, pointB.hidden_count])
#
#     return possible_cuts