# # import numpy as np
# from ann_point.Functions import *
# # from ann_point.AnnPoint2 import *
# from evolving_classifier.operators.PuzzleCO import PuzzleCO
# from evolving_classifier.operators.CrossoverOperator import *
# # from utility.Mut_Utility import resize_layer
# from evolving_classifier.operators.CrossoverOperator2 import FinalCrossoverOperator2
# from utility.TestingUtility import assert_chaos_network_properties
#
# # TODO - B - test this if used
# def test_simple_crossover():
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
#                                  dstr_mut_prob=(0, 0)) # values irrelevant aside from neuron count
#
#
#     link1 = np.ones((15, 15))
#     link1 = np.multiply(link1, get_weight_mask(2, 3, 15))
#     wei1 = np.zeros((link1.shape))
#
#     nonz = np.where(link1 != 0)
#     n = 1001
#     for i in range(len(nonz[0])):
#         wei1[nonz[0][i], nonz[1][i]] = n
#         n += 1
#
#     bia1 = np.array([[0., 0, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]])
#     actFuns1 = [None, None, ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), None, None, None]
#
#     link2 = np.multiply(np.ones((10, 10)), get_weight_mask(2, 3, 10))
#     wei2 = np.zeros((link2.shape))
#
#     nonz = np.where(link2 != 0)
#     n = 2001
#     for i in range(len(nonz[0])):
#         wei2[nonz[0][i], nonz[1][i]] = n
#         n += 1
#     bia2 = np.array([[0., 0, 203, 204, 205, 206, 207, 208, 209, 210]])
#     actFuns2 = [None, None, TanH(), Poly3(), TanH(), Poly3(), TanH(), None, None, None]
#
#     cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
#                    aggrFun=SincAct(), net_it=2, mutation_radius=1, sqr_mut_prob=2,
#                    lin_mut_prob=3, p_mutation_prob=4, c_prob=5, dstr_mut_prob=6)
#     cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
#                    aggrFun=GaussAct(), net_it=5, mutation_radius=10, sqr_mut_prob=20,
#                    lin_mut_prob=30, p_mutation_prob=40, c_prob=50, dstr_mut_prob=60)
#
#     co = PuzzleCO(hrange)
#
#     random.seed(1001)
#     np.random.seed(1001)
#     cn3, cn4 = co.crossover(cn1, cn2)
#
#
#     wei3 = np.zeros((11, 11))
#     wei3[0:8, 2:7] = wei1[0:8, 2:7]
#     wei3[0:7, 7] = wei2[0:7, 2]
#     wei3[2:7, -3:] = wei1[2:7, -3:]
#     wei3[7, -3:] = wei2[2, -3:]
#
#     wei4 = np.zeros((12, 12))
#     wei4[0:9, 2:8] = wei1[0:9, 3:9]
#     wei4[2:8, -3:] = wei1[3:9, -3:]
#     wei4[0:7, 8] = wei2[0:7, 6]
#     wei4[8, -3:] = wei2[6, -3:]
#     wei4 = np.multiply(wei4, get_weight_mask(2, 3, 12))
#
#     assert_chaos_network_properties(net=cn3,
#                           desired_input_size=2,
#                           desited_output_size=3,
#                           desired_neuron_count=11,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=8,
#                           desired_hidden_count=6,
#                           desired_links=np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
#                                                   [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#                           desired_weights=wei3,
#                           desired_biases=np.array([[0, 0, 103, 104, 105, 106, 107, 203, 208, 114, 210]]),
#                           desired_actFun=[None, None, ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), TanH(), None, None, None],
#                           desired_aggr=SincAct(),
#                           desired_maxit=5,
#                           desired_mut_rad=1,
#                           desired_wb_prob=2,
#                           desired_s_prob=30,
#                           desired_p_prob=4,
#                           desired_c_prob=5,
#                           desired_r_prob=6)
#
#     ##################################################################
#
#     assert_chaos_network_properties(net=cn4,
#                           desired_input_size=2,
#                           desited_output_size=3,
#                           desired_neuron_count=12,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=9,
#                           desired_hidden_count=7,
#                           desired_links=np.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
#                                                   [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#                           desired_weights=wei4,
#                           desired_biases=np.array([[0., 0, 104, 105, 106, 107, 108, 109, 207, 208, 114, 115]]),
#                           desired_actFun=[None, None, Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), TanH(), None, None, None],
#                           desired_aggr=GaussAct(),
#                           desired_maxit=2,
#                           desired_mut_rad=10,
#                           desired_wb_prob=20,
#                           desired_s_prob=3,
#                           desired_p_prob=40,
#                           desired_c_prob=50,
#                           desired_r_prob=60)
#
#
# def test_simple_crossover_2():
#
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
#                                  dstr_mut_prob=(0, 0)) # values irrelevant aside from neuron count
#
#
#     link1 = np.ones((6, 6))
#     link1 = np.multiply(link1, get_weight_mask(2, 3, 6))
#     wei1 = np.zeros((link1.shape))
#
#     nonz = np.where(link1 != 0)
#     n = 1001
#     for i in range(len(nonz[0])):
#         wei1[nonz[0][i], nonz[1][i]] = n
#         n += 1
#
#     bia1 = np.array([[0., 0, 103, 104, 105, 106]])
#     actFuns1 = [None, None, ReLu(), None, None, None]
#
#     link2 = np.multiply(np.ones((10, 10)), get_weight_mask(2, 3, 10))
#     wei2 = np.zeros((link2.shape))
#
#     nonz = np.where(link2 != 0)
#     n = 2001
#     for i in range(len(nonz[0])):
#         wei2[nonz[0][i], nonz[1][i]] = n
#         n += 1
#     bia2 = np.array([[0., 0, 203, 204, 205, 206, 207, 208, 209, 210]])
#     actFuns2 = [None, None, TanH(), Poly3(), TanH(), Poly3(), TanH(), None, None, None]
#
#     cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
#                    aggrFun=SincAct(), net_it=2, mutation_radius=1, sqr_mut_prob=2,
#                    lin_mut_prob=3, p_mutation_prob=4, c_prob=5, dstr_mut_prob=6)
#     cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
#                    aggrFun=GaussAct(), net_it=5, mutation_radius=10, sqr_mut_prob=20,
#                    lin_mut_prob=30, p_mutation_prob=40, c_prob=50, dstr_mut_prob=60)
#
#     co = FinalCrossoverOperator2(hrange)
#
#     random.seed(23232323)
#     np.random.seed(23232323)
#     cn3, cn4 = co.crossover(cn1, cn2)
#
#
#     wei3 = np.zeros((8, 8))
#     wei3[0:3, 2] = wei1[0:3, 2]
#     wei3[0:5, 3:5] = wei2[0:5, 2:4]
#     wei3[2, -3:] = wei1[2, -3:]
#     wei3[3:5, -3:] = wei2[2:4, -3:]
#     wei3 = np.multiply(wei3, get_weight_mask(2, 3, 8))
#
#     wei4 = np.zeros((9, 9))
#     wei4[0:3, 2] = wei1[0:3, 2]
#     wei4[2, -3:] = wei1[2, -3:]
#     wei4[0:7, 3:6] = wei2[0:7, 3:6]
#     wei4[3:6, -3:] = wei2[3:6, -3:]
#     wei4 = np.multiply(wei4, get_weight_mask(2, 3, 9))
#
#     assert_chaos_network_properties(net=cn3,
#                           desired_input_size=2,
#                           desited_output_size=3,
#                           desired_neuron_count=8,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=5,
#                           desired_hidden_count=3,
#                           desired_links=np.array([[0, 0, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 0, 0, 1, 1, 1, 1],
#                                                   [0, 0, 0, 0, 0, 1, 1, 1],
#                                                   [0, 0, 0, 1, 0, 1, 1, 1],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0]]),
#                           desired_weights=wei3,
#                           desired_biases=np.array([[0, 0, 103, 203, 204, 104, 105, 210]]),
#                           desired_actFun=[None, None, ReLu(), TanH(), Poly3(), None, None, None],
#                           desired_aggr=SincAct(),
#                           desired_maxit=2,
#                           desired_mut_rad=1,
#                           desired_wb_prob=20,
#                           desired_s_prob=30,
#                           desired_p_prob=40,
#                           desired_c_prob=5,
#                           desired_r_prob=6)
#
#     ##################################################################
#
#     assert_chaos_network_properties(net=cn4,
#                           desired_input_size=2,
#                           desited_output_size=3,
#                           desired_neuron_count=9,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=6,
#                           desired_hidden_count=4,
#                           desired_links=np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 1, 1, 1, 1, 0, 0, 0],
#                                                   [0, 0, 0, 1, 1, 1, 1, 1, 1],
#                                                   [0, 0, 0, 0, 1, 1, 1, 1, 1],
#                                                   [0, 0, 0, 1, 0, 1, 1, 1, 1],
#                                                   [0, 0, 0, 1, 1, 0, 1, 1, 1],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
#                           desired_weights=wei4,
#                           desired_biases=np.array([[0, 0, 103, 204, 205, 206, 104, 209, 106]]),
#                           desired_actFun=[None, None, ReLu(), Poly3(), TanH(), Poly3(), None, None, None],
#                           desired_aggr=GaussAct(),
#                           desired_maxit=5,
#                           desired_mut_rad=10,
#                           desired_wb_prob=2,
#                           desired_s_prob=3,
#                           desired_p_prob=4,
#                           desired_c_prob=50,
#                           desired_r_prob=60)
#
# # random.seed(1001)
# # np.random.seed(1001)
# # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
# #                              sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
# #                              dstr_mut_prob=(0, 0)) # values irrelevant aside from neuron count
# #
# #
# # link1 = np.ones((15, 15))
# # link1 = np.multiply(link1, get_weight_mask(2, 3, 15))
# # wei1 = np.zeros((link1.shape))
# #
# # nonz = np.where(link1 != 0)
# # n = 1001
# # for i in range(len(nonz[0])):
# #     wei1[nonz[0][i], nonz[1][i]] = n
# #     n += 1
# #
# # bia1 = np.array([[0., 0, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]])
# # actFuns1 = [None, None, ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), ReLu(), Poly2(), None, None, None]
# #
# # link2 = np.multiply(np.ones((10, 10)), get_weight_mask(2, 3, 10))
# # wei2 = np.zeros((link2.shape))
# #
# # nonz = np.where(link2 != 0)
# # n = 2001
# # for i in range(len(nonz[0])):
# #     wei2[nonz[0][i], nonz[1][i]] = n
# #     n += 1
# # bia2 = np.array([[0., 0, 203, 204, 205, 206, 207, 208, 209, 210]])
# # actFuns2 = [None, None, TanH(), Poly3(), TanH(), Poly3(), TanH(), None, None, None]
# #
# # cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
# #                aggrFun=SincAct(), net_it=2, mutation_radius=1, sqr_mut_prob=2,
# #                lin_mut_prob=3, p_mutation_prob=4, c_prob=5, dstr_mut_prob=6)
# # cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
# #                aggrFun=GaussAct(), net_it=5, mutation_radius=10, sqr_mut_prob=20,
# #                lin_mut_prob=30, p_mutation_prob=40, c_prob=50, dstr_mut_prob=60)
# #
# # pc = find_possible_cuts4(cn1, cn2, hrange)
# # cuts = choose_without_repetition(pc, 2)
# # print(cuts[0])
# # print(cuts[1])
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print("---")
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print("---")
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# #
# # test_simple_crossover()
#
# # seed = 23232323
# # random.seed(seed)
# # np.random.seed(seed)
# #
# # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
# #                              wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
# #                              r_prob=(0, 0)) # values irrelevant aside from neuron count
# #
# #
# # link1 = np.ones((6, 6))
# # link1 = np.multiply(link1, get_weight_mask(2, 3, 6))
# # wei1 = np.zeros((link1.shape))
# #
# # nonz = np.where(link1 != 0)
# # n = 1001
# # for i in range(len(nonz[0])):
# #     wei1[nonz[0][i], nonz[1][i]] = n
# #     n += 1
# #
# # bia1 = np.array([[0., 0, 103, 104, 105, 106]])
# # actFuns1 = [None, None, ReLu(), None, None, None]
# #
# # link2 = np.multiply(np.ones((10, 10)), get_weight_mask(2, 3, 10))
# # wei2 = np.zeros((link2.shape))
# #
# # nonz = np.where(link2 != 0)
# # n = 2001
# # for i in range(len(nonz[0])):
# #     wei2[nonz[0][i], nonz[1][i]] = n
# #     n += 1
# # bia2 = np.array([[0., 0, 203, 204, 205, 206, 207, 208, 209, 210]])
# # actFuns2 = [None, None, TanH(), Poly3(), TanH(), Poly3(), TanH(), None, None, None]
# #
# # cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
# #                aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2,
# #                s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
# # cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
# #                aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20,
# #                s_mutation_prob=30, p_mutation_prob=40, c_prob=50, r_prob=60)
# #
# # pc = find_possible_cuts4(cn1, cn2, hrange)
# # cuts = choose_without_repetition(pc, 2)
# # print(cuts[0])
# # print(cuts[1])
# #
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print("---")
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print("---")
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# # print(random.random())
# #
# # test_simple_crossover_2()
#
