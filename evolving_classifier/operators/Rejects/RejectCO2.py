# import random
#
# from ann_point import HyperparameterRange
# from evolving_classifier.operators.Rejects.FinalCO1 import find_possible_cuts4
# from neural_network.ChaosNet import ChaosNet
# import numpy as np
#
# from utility.Mut_Utility import conditional_value_swap, get_weight_mask
# from utility.Utility import choose_without_repetition
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
#
# class RejectCO2(CrossoverOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__()
#         self.hrange = hrange
#
#     def crossover(self, pointA: ChaosNet, pointB: ChaosNet) -> [ChaosNet, ChaosNet]:
#         possible_cuts = find_possible_cuts4(pointA, pointB, self.hrange)
#
#         cuts = choose_without_repetition(options=possible_cuts, count=2)
#
#         # if len(cuts) == 1: # obie sieci mają zero neuronów ukrytych
#         #     cuts.append([1, 0, 1, 0])
#
#         input_size = pointA.input_size
#         output_size = pointA.output_size
#
#         new_A_links, new_A_weights, new_A_biases, new_A_func = get_link_weights_biases_acts4(pointA=pointA, pointB=pointB, cut=cuts[0])
#         new_B_links, new_B_weights, new_B_biases, new_B_func = get_link_weights_biases_acts4(pointA=pointA, pointB=pointB, cut=cuts[1])
#
#         # cut_A = cut[1]
#         # cut_B = cut[2]
#         # A_1 = cut[3]
#         # A_2 = cut[4]
#         # B_1 = cut[5]
#         # B_2 = cut[6]
#         #
#         # new_A_hidden_count = A_1 + B_2
#         # new_B_hidden_count = A_2 + B_1
#         # new_A_count = pointA.input_size + new_A_hidden_count + pointA.output_size
#         # new_B_count = pointB.input_size + new_B_hidden_count + pointB.output_size
#         #
#         # rows_to_copy_A_A = min(pointA.hidden_end_index, new_A_count - output_size)
#         # rows_to_copy_A_B = min(pointA.hidden_end_index, new_B_count - output_size)
#         # rows_to_copy_B_A = min(pointB.hidden_end_index, new_A_count - output_size)
#         # rows_to_copy_B_B = min(pointB.hidden_end_index, new_B_count - output_size)
#         #
#         # # link swap
#         # new_A_links = np.zeros((new_A_count, new_A_count))
#         # new_B_links = np.zeros((new_B_count, new_B_count))
#         #
#         # new_A_links[:rows_to_copy_A_A, :cut_A] = pointA.links[:rows_to_copy_A_A, :cut_A]
#         # new_A_links[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.links[:rows_to_copy_B_A, -(output_size + B_2):]
#         #
#         # new_B_links[:rows_to_copy_B_B, :cut_B] = pointB.links[:rows_to_copy_B_B, :cut_B]
#         # new_B_links[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.links[:rows_to_copy_A_B, -(output_size + A_2):]
#         #
#         # # weight swap
#         # new_A_weights = np.zeros((new_A_count, new_A_count))
#         # new_B_weights = np.zeros((new_B_count, new_B_count))
#         #
#         # new_A_weights[:rows_to_copy_A_A, :cut_A] = pointA.weights[:rows_to_copy_A_A, :cut_A]
#         # new_A_weights[:rows_to_copy_B_A, -(output_size + B_2):] = pointB.weights[:rows_to_copy_B_A, -(output_size + B_2):]
#         #
#         # new_B_weights[:rows_to_copy_B_B, :cut_B] = pointB.weights[:rows_to_copy_B_B, :cut_B]
#         # new_B_weights[:rows_to_copy_A_B, -(output_size + A_2):] = pointA.weights[:rows_to_copy_A_B, -(output_size + A_2):]
#         #
#         # # bias swap
#         # new_A_biases = np.zeros((1, new_A_count))
#         # new_B_biases = np.zeros((1, new_B_count))
#         #
#         # new_A_biases[0, :cut_A] = pointA.biases[0, :cut_A]
#         # new_A_biases[0, -(output_size + B_2):] = pointB.biases[0, -(output_size + B_2):]
#         #
#         # new_B_biases[0, :cut_B] = pointB.biases[0, :cut_B]
#         # new_B_biases[0, -(output_size + A_2):] = pointA.biases[0, -(output_size + A_2):]
#         #
#         # # actFun swap
#         # new_A_func = input_size * [None]
#         # new_B_func = input_size * [None]
#         #
#         # for i in range(A_1):
#         #     new_A_func.append(pointA.actFuns[input_size + i].copy())
#         # for i in range(B_2):
#         #     new_A_func.append(pointB.actFuns[input_size + B_1 + i].copy())
#         # for i in range(output_size):
#         #     new_A_func.append(None)
#         #
#         # for i in range(B_1):
#         #     new_B_func.append(pointB.actFuns[input_size + i].copy())
#         # for i in range(A_2):
#         #     new_B_func.append(pointA.actFuns[input_size + A_1 + i].copy())
#         # for i in range(output_size):
#         #     new_B_func.append(None)
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
# def cut_into_puzzles4(matrix: np.ndarray, i: int, o: int, start: int, num: int) -> [np.ndarray]:
#     # if left:
#     #     P1 = matrix[i:-o, start:start+num]
#     #     P2 = matrix[:i, start:start+num]
#     #     P3 = matrix[i:-o, -o:]
#     # else:
#     #     P1 = matrix[i:-o, start:start+num]
#     #     P2 = matrix[:i, start:start+num]
#     #     P3 = matrix[i:-o, -o:]
#
#     P1 = matrix[i:-o, start:start+num]
#     P2 = matrix[:i, start:start+num]
#     P3 = matrix[i:-o, -o:]
#
#     return P1, P2, P3
#
# def piece_together_from_puzzles4(i: int, o: int, left_puzzles: [np.ndarray], right_puzzles: [np.ndarray]):
#     left_nc = left_puzzles[0].shape[1]
#     right_nc = right_puzzles[0].shape[1]
#
#     n = i + o + left_nc + right_nc
#     hei = i + left_nc + right_nc
#     aEnd = i + left_nc
#
#     result = np.zeros((n, n))
#
#     # Put in pieces #1
#     l1h = min(left_puzzles[0].shape[0], hei-i)
#     result[i:i+l1h, i:aEnd] = left_puzzles[0][:l1h, :]
#
#     r1h = min(right_puzzles[0].shape[0], hei-i)
#     result[-(r1h + o):-o, aEnd:-o] = right_puzzles[0][-r1h:, :]
#
#     # Put in pieces #2
#     result[:i, i:aEnd] = left_puzzles[1]
#     result[:i, aEnd:-o] = right_puzzles[1]
#
#     # Put in pieces #3
#     result[i:i+l1h, -o:] = left_puzzles[2][:l1h, :]
#     result[-(r1h + o):-o, -o:] = right_puzzles[2][-r1h, :]
#
#     return result
#
# def get_link_weights_biases_acts4(pointA: ChaosNet, pointB: ChaosNet, cut: [int]):
#     input_size = pointA.input_size
#     output_size = pointA.output_size
#
#     if cut[1] != 0:
#         lA = cut[0] - pointA.hidden_start_index
#         rA = pointA.hidden_end_index - cut[0] - cut[1]
#     else:
#         lA = 0
#         rA = 0
#
#     if cut[3] != 0:
#         lB = cut[2] - pointB.hidden_start_index
#         rB = pointB.hidden_end_index - cut[2] - cut[3]
#     else:
#         lB = 0
#         rB = 0
#
#     # if lA + rB > rA + lB:
#     #     tmp = pointA
#     #     pointA = pointB
#     #     pointB = tmp
#     #
#     #     tmp = cut[0]
#     #     cut[0] = cut[2]
#     #     cut[2] = tmp
#     #
#     #     tmp = cut[1]
#     #     cut[1] = cut[3]
#     #     cut[3] = tmp
#
#
#     links = piece_together_from_puzzles4(i=input_size, o=output_size,
#                                          left_puzzles=cut_into_puzzles4(matrix=pointA.links, i=input_size, o=output_size,
#                                                                         start=cut[0], num=cut[1]),
#                                          right_puzzles=cut_into_puzzles4(matrix=pointB.links, i=input_size, o=output_size,
#                                                                          start=cut[2], num=cut[3]))
#     links = np.multiply(links, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))
#
#     weights = piece_together_from_puzzles4(i=input_size, o=output_size,
#                                            left_puzzles=cut_into_puzzles4(matrix=pointA.weights, i=input_size, o=output_size,
#                                                                           start=cut[0], num=cut[1]),
#                                            right_puzzles=cut_into_puzzles4(matrix=pointB.weights, i=input_size, o=output_size,
#                                                                            start=cut[2], num=cut[3]))
#     weights = np.multiply(weights, get_weight_mask(pointA.input_size, pointA.output_size, links.shape[0]))
#     nc = input_size + output_size + cut[1] + cut[3]
#
#     biases = np.zeros((1, nc))
#     acts = nc * [None]
#     for i in range(cut[1]):
#         biases[0, input_size + i] = pointA.biases[0, cut[0] + i].copy()
#         acts[input_size + i] = pointA.actFuns[cut[0] + i]
#     for i in range(cut[3]):
#         biases[0, input_size + cut[1] + i] = pointB.biases[0, cut[2] + i]
#         acts[input_size + cut[1] + i] = pointB.actFuns[cut[2] + i].copy()
#
#     for i in range(output_size):
#         swap = random.random()
#         if swap <= 0.5:
#             biases[0, -output_size + i] = pointA.biases[0, -output_size + i]
#         else:
#             biases[0, -output_size + i] = pointB.biases[0, -output_size + i]
#
#     return links, weights, biases, acts
