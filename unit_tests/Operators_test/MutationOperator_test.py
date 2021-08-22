# import random
# import numpy as np
# # import pytest
#
# # from ann_point.AnnPoint2 import AnnPoint2
# from ann_point.Functions import *
# from ann_point.HyperparameterRange import HyperparameterRange
# from evolving_classifier.operators.MutationOperators import *
# from utility.Mut_Utility import *
# from utility.TestingUtility import compare_chaos_network
#
# # TODO - A - TEST!!!
# def test_struct_mutation():
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.8, 1), c_prob=(0.22, 0.33),
#                                  dstr_mut_prob=(0.44, 0.55))
#     mo = FinalMutationOperator(hrange)
#
#     random.seed(20021234)
#     np.random.seed(20021234)
#
#     link1 = np.array([[0, 1, 1, 0, 1],
#                       [0, 0, 1, 0, 1],
#                       [0, 1, 0, 0, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     wei1 = np.array([[0., 1, 2, 0, 4],
#                      [0, 0, 3, 0, 5],
#                      [0, 7, 0, 0, 6],
#                      [0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0]])
#     bia1 = np.array([[0., -2, -3, -4, -5]])
#     actFuns1 = [None, ReLu(), TanH(), None, None]
#
#     cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=TanH(), net_it=2, mutation_radius=0.5, sqr_mut_prob=0.5,
#                    lin_mut_prob=0.8, p_mutation_prob=0.7, c_prob=0.3, dstr_mut_prob=0.6)
#
#     mutant = mo.mutate(cn1)
#
#     compare_chaos_network(net=cn1,
#                           desired_input_size=1,
#                           desited_output_size=2,
#                           desired_neuron_count=5,
#                           desired_hidden_start_index=1,
#                           desired_hidden_end_index=3,
#                           desired_hidden_count=2,
#                           desired_links=np.array([[0, 1, 1, 0, 1],
#                                                   [0, 0, 1, 0, 1],
#                                                   [0, 1, 0, 0, 1],
#                                                   [0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0]]),
#                           desired_weights=np.array([[0., 1, 2, 0, 4],
#                                                     [0, 0, 3, 0, 5],
#                                                     [0, 7, 0, 0, 6],
#                                                     [0, 0, 0, 0, 0],
#                                                     [0, 0, 0, 0, 0]]),
#                           desired_biases=np.array([[0., -2, -3, -4, -5]]),
#                           desired_actFun=[None, ReLu(), TanH(), None, None],
#                           desired_aggr=TanH(),
#                           desired_maxit=2,
#                           desired_mut_rad=0.5,
#                           desired_wb_prob=0.5,
#                           desired_s_prob=0.8,
#                           desired_p_prob=0.7,
#                           desired_c_prob=0.3,
#                           desired_r_prob=0.6)
#
#
#     # wb_pm = 0.5
#     # s_pm = 0.8
#     # p_pm = 0.7
#     # r_pm = 0.4
#     # c_pm = 0.3
#
#     compare_chaos_network(net=mutant,
#                           desired_input_size=1,
#                           desited_output_size=2,
#                           desired_neuron_count=7,
#                           desired_hidden_start_index=1,
#                           desired_hidden_end_index=5,
#                           desired_hidden_count=4,
#                           desired_links=np.array([[0, 0, 0, 1, 1, 0, 0],
#                                                   [0, 0, 0, 1, 1, 1, 0],
#                                                   [0, 0, 0, 1, 1, 0, 1],
#                                                   [0, 1, 0, 0, 1, 1, 1],
#                                                   [0, 0, 0, 1, 0, 1, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0, 0, 0]]),
#                           desired_weights=np.array([[0, 0,          0,          1.87464534, 3.9600569 , 0         , 0.        ],
#                                                     [0, 0,          0,          1.68673925, 1.75851153, 1.7610361 , 0.        ],
#                                                     [0, 0,          0,          4.83071027, 3.37009655, 0.        , 6.21459144],
#                                                     [0, 1.03066458, 0.        , 0.        , 0.29136715, 7.38647365, 0.47869675],
#                                                     [0.,0.,         0.,         2.69646178, 0.        , 6.01084322, 0.        ],
#                                                     [0, 0, 0, 0, 0, 0, 0],
#                                                     [0, 0, 0, 0, 0, 0, 0]]),
#                           desired_biases=np.array([[0, -2.         ,-2.29397386, -0.5077799,  -2.38556375, -3.47196718, -2.86713405]]),
#                           desired_actFun=[None, GaussAct(), GaussAct(), Sigmoid(), ReLu(), None, None],
#                           desired_aggr=GaussAct(),
#                           desired_maxit=3,
#                           desired_mut_rad=0.5,
#                           desired_wb_prob=0.085585,
#                           desired_s_prob=0.604876,
#                           desired_p_prob=0.879561,
#                           desired_c_prob=0.3,
#                           desired_r_prob=0.6)
#
# def test_struct_mutation_2():
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.8, 1), c_prob=(0.22, 0.33),
#                                  dstr_mut_prob=(0.44, 0.55))
#     mo = FinalMutationOperator(hrange)
#
#     random.seed(1006)
#     np.random.seed(1006)
#
#     link1 = np.array([[0, 1, 1, 0, 1],
#                       [0, 0, 1, 0, 1],
#                       [0, 1, 0, 0, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     wei1 = np.array([[0., 1, 2, 0, 4],
#                      [0, 0, 3, 0, 5],
#                      [0, 7, 0, 0, 6],
#                      [0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 0]])
#     bia1 = np.array([[0., -2, -3, -4, -5]])
#     actFuns1 = [None, ReLu(), ReLu(), None, None]
#     #TODO - B - nie ma sensu żeby przekazywać prawd do operatora bo są w punkcie i tak
#     cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
#                    aggrFun=TanH(), net_it=2, mutation_radius=0.5, sqr_mut_prob=0.4, lin_mut_prob=0.5, p_mutation_prob=0.4,
#                    c_prob=0.9, dstr_mut_prob=0.75)
#
#     mutant = mo.mutate(cn1)
#
#
#
#     compare_chaos_network(net=cn1,
#                           desired_input_size=1,
#                           desited_output_size=2,
#                           desired_neuron_count=5,
#                           desired_hidden_start_index=1,
#                           desired_hidden_end_index=3,
#                           desired_hidden_count=2,
#                           desired_links=np.array([[0, 1, 1, 0, 1],
#                                                   [0, 0, 1, 0, 1],
#                                                   [0, 1, 0, 0, 1],
#                                                   [0, 0, 0, 0, 0],
#                                                   [0, 0, 0, 0, 0]]),
#                           desired_weights=np.array([[0., 1, 2, 0, 4],
#                                                     [0, 0, 3, 0, 5],
#                                                     [0, 7, 0, 0, 6],
#                                                     [0, 0, 0, 0, 0],
#                                                     [0, 0, 0, 0, 0]]),
#                           desired_biases=np.array([[0., -2, -3, -4, -5]]),
#                           desired_actFun=[None, ReLu(), ReLu(), None, None],
#                           desired_aggr=TanH(),
#                           desired_maxit=2,
#                           desired_mut_rad=0.5,
#                           desired_wb_prob=0.4,
#                           desired_s_prob=0.5,
#                           desired_p_prob=0.4,
#                           desired_c_prob=0.9,
#                           desired_r_prob=0.75)
#
#     compare_chaos_network(net=mutant,
#                           desired_input_size=1,
#                           desited_output_size=2,
#                           desired_neuron_count=3,
#                           desired_hidden_start_index=1,
#                           desired_hidden_end_index=1,
#                           desired_hidden_count=0,
#                           desired_links=np.array([[0., 0., 0.],
#                                                   [0., 0., 0.],
#                                                   [0., 0., 0.]]),
#                           desired_weights=np.array([[0., 0., 0.],
#                                                     [0., 0., 0.],
#                                                     [0., 0., 0.]]),
#                           desired_biases=np.array([[-0.        , -4.51179213, -2.61077568]]),
#                           desired_actFun=[None, None, None],
#                           desired_aggr=TanH(),
#                           desired_maxit=2,
#                           desired_mut_rad=0.5,
#                           desired_wb_prob=0.4,
#                           desired_s_prob=0.600238,
#                           desired_p_prob=0.859729,
#                           desired_c_prob=0.9,
#                           desired_r_prob=0.75)
#
# def test_struct_mutation_3():
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.8, 1), c_prob=(0.22, 0.33),
#                                  dstr_mut_prob=(0.44, 0.55))
#     mo = FinalMutationOperator(hrange)
#
#     random.seed(1009)
#     np.random.seed(1009)
#
#     link1 = np.array([[0, 0, 1, 1, 0],
#                       [0, 0, 1, 0, 0],
#                       [0, 0, 0, 1, 1],
#                       [0, 0, 1, 0, 1],
#                       [0, 0, 0, 0, 0]])
#     wei1 = np.array([[0, 0, 1, 4, 0],
#                      [0, 0, 2, 0, 0],
#                      [0, 0, 0, 5, 8],
#                      [0, 0, 3, 0, 9],
#                      [0, 0, 0, 0, 0.]])
#     bia1 = np.array([[0., 0, -3, -4, -5]])
#     actFuns1 = [None, None, TanH(), Sigmoid(), None]
#
#     cn1 = ChaosNet(input_size=2, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=TanH(), net_it=4, mutation_radius=1.2,
#                    sqr_mut_prob=0.2, lin_mut_prob=0.8, p_mutation_prob=1., c_prob=0.3, dstr_mut_prob=0.35)
#
#     mutant = mo.mutate(cn1)
#
#     compare_chaos_network(net=cn1,
#                           desired_input_size=2,
#                           desited_output_size=1,
#                           desired_neuron_count=5,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=4,
#                           desired_hidden_count=2,
#                           desired_links=np.array([[0, 0, 1, 1, 0],
#                                                   [0, 0, 1, 0, 0],
#                                                   [0, 0, 0, 1, 1],
#                                                   [0, 0, 1, 0, 1],
#                                                   [0, 0, 0, 0, 0]]),
#                           desired_weights=np.array([[0, 0, 1, 4, 0],
#                                                     [0, 0, 2, 0, 0],
#                                                     [0, 0, 0, 5, 8],
#                                                     [0, 0, 3, 0, 9],
#                                                     [0, 0, 0, 0, 0.]]),
#                           desired_biases=np.array([[0., 0, -3, -4, -5]]),
#                           desired_actFun=[None, None, TanH(), Sigmoid(), None],
#                           desired_aggr=TanH(),
#                           desired_maxit=4,
#                           desired_mut_rad=1.2,
#                           desired_wb_prob=0.2,
#                           desired_s_prob=0.8,
#                           desired_p_prob=1,
#                           desired_c_prob=0.3,
#                           desired_r_prob=0.35)
#
#     compare_chaos_network(net=mutant,
#                           desired_input_size=2,
#                           desited_output_size=1,
#                           desired_neuron_count=5,
#                           desired_hidden_start_index=2,
#                           desired_hidden_end_index=4,
#                           desired_hidden_count=2,
#                           desired_links=np.array([[0., 0., 1., 0., 0.],
#                                                   [0., 0., 0., 1., 0.],
#                                                   [0., 0., 0., 1., 0.],
#                                                   [0., 0., 0., 0., 0.],
#                                                   [0., 0., 0., 0., 0.]]),
#                           desired_weights=np.array([[0, 0, 7.29343795, 0, 0],
#                                                     [0, 0, 0, 5.98317628, 0],
#                                                     [0, 0, 0, 5, 0],
#                                                     [0, 0, 0, 0, 0],
#                                                     [0, 0, 0, 0, 0.]]),
#                           desired_biases=np.array([[ 0.,         -0.,         -3.,         -3.78908424, -3.49521686]]),
#                           desired_actFun=[None, None, Sigmoid(), Sigmoid(), None],
#                           desired_aggr=Sigmoid(),
#                           desired_maxit=4,
#                           desired_mut_rad=0.8160267,
#                           desired_wb_prob=0.0607029,
#                           desired_s_prob=0.620329,
#                           desired_p_prob=0.934746,
#                           desired_c_prob=0.30211,
#                           desired_r_prob=0.491017)
#
# # seed = 1009
# # random.seed(seed)
# # np.random.seed(seed)
# # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 2),
# #                              wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.8, 1), c_prob=(0.22, 0.33),
# #                              r_prob=(0.44, 0.55))
# # link1 = np.array([[0, 0, 1, 1, 1],
# #                   [0, 0, 1, 0, 1],
# #                   [0, 0, 0, 1, 1],
# #                   [0, 0, 1, 0, 1],
# #                   [0, 0, 0, 0, 0]])
# # wei1 = np.array([[0, 0, 1, 4, 6],
# #                  [0, 0, 2, 0, 7],
# #                  [0, 0, 0, 5, 8],
# #                  [0, 0, 3, 0, 9],
# #                  [0, 0, 0, 0, 0.]])
# # bia1 = np.array([[0., 0, -3, -4, -5]])
# # act1 = [None, None, TanH(), Sigmoid(), None]
# # aggr1 = TanH()
# # maxit = 4
# # input_size = 2
# # hidden_size = 2
# # output_size = 1
# # neuron_count = 5
# # radius = 1.2
# # frac_radius = radius / hrange.max_mut_radius
# #
# # wb_pm = 0.2
# # s_pm = 0.8
# # p_pm = 1
# # c_pm = 0.3
# # r_pm = 0.35
# #
# #
# # print(wei1)
# # print(bia1)
# #
# # wei2 = gaussian_shift(wei1, link1, wb_pm, radius).copy()
# # print(f"wei_shifted: \n{wei2}")
# #
# # wei3 = reroll_matrix(wei2, link1, r_pm, 1, np.max(wei2)).copy()
# # print(f"wei_rerolled: \n{wei3}")
# #
# # bia2 = gaussian_shift(bia1, get_bias_mask(input_size, neuron_count), wb_pm, radius).copy()
# # print(f"bia_shifted: \n{bia2}")
# #
# # bia3 = reroll_matrix(bia2, get_bias_mask(input_size=input_size, neuron_count=neuron_count), r_pm, -5, -3).copy()
# # print(f"bia_rerolled: \n{bia3}")
# #
# # for i in range(input_size, input_size+hidden_size):
# #     if random.random() <= s_pm:
# #         tmp_act = try_choose_different(act1[i], hrange.actFunSet)
# #         print(f"act {i}: \n{tmp_act.to_string()}")
# #         act1[i] = tmp_act
# #
# #
# # print(f"aggr: \n{conditional_try_choose_different(s_pm, aggr1, hrange.actFunSet).to_string()}")
# #
# # point = ChaosNet(input_size=input_size, output_size=output_size, links=link1, weights=wei3,
# #                   biases=bia3, actFuns=act1, aggrFun=TanH(), maxit=maxit, mutation_radius=radius,
# #                   wb_mutation_prob=wb_pm, s_mutation_prob=s_pm, p_mutation_prob=p_pm, c_prob=c_pm, r_prob=r_pm)
# #
# # nc = conditional_try_choose_different(s_pm, 2, [0, 1, 2, 3, 4])
# # point = change_neuron_count(net=point, hrange=hrange, demanded_hidden=nc).copy()
# #
# # wei4 = point.weights.copy()
# # bia4 = point.biases.copy()
# # link2 = point.links.copy()
# # act2 = point.actFuns.copy()
# #
# # print(f"wei_changed_count:\n{wei4}")
# # print(f"bia_changed_count:\n{bia4}")
# # for i in range(point.input_size, point.hidden_end_index):
# #     print(f"act_2_{i}:\n{act2[i].to_string()}")
# #
# # wei5, link3 = add_remove_weights(s_pm, wei4, link2, get_weight_mask(point.input_size, point.output_size, point.neuron_count))
# #
# # wei5 = wei5.copy()
# # link3 = link3.copy()
# #
# # print(f"weights_ar:\n{wei5}")
# # print(f"links_ar:\n{link3}")
# # print(f"is:\n{point.input_size}")
# # print(f"os:\n{point.output_size}")
# # print(f"nc:\n{point.neuron_count}")
# #
# # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
# #                              wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.8, 1), c_prob=(0.22, 0.33),
# #                              r_prob=(0.44, 0.55))
# #
# # print(f"max_it:\n {conditional_try_choose_different(s_pm, maxit, [1, 2, 3, 4, 5])}")
# #
# #
# # print(f"rad:\n{reroll_value(p_pm, radius, 0, 1)}")
# # print(f"wb:\n{reroll_value(p_pm, wb_pm, 0.05, 0.1)}")
# # print(f"s:\n{reroll_value(p_pm, s_pm, 0.6, 0.7)}")
# # print(f"p:\n{reroll_value(p_pm, p_pm, 0.8, 1)}")
# # print(f"c:\n{reroll_value(p_pm, c_pm, 0.22, 0.33)}")
# # print(f"r:\n{reroll_value(p_pm, r_pm, 0.44, 0.55)}")
# #
# # # test_struct_mutation()
# # # test_struct_mutation_2()
# # test_struct_mutation_3()
#
#
#
# # ch1 = np.zeros((neuron_count, neuron_count))
# # ch1[np.random.random((neuron_count, neuron_count)) <= 0.75] = 1
# # ch1[:, :input_size] = 0
# # ch1[-output_size:, :] = 0
# # np.fill_diagonal(ch1, 0)
# # # print(f"change_prob: \n {ch1}")
# # where_should_weights_change = np.where(ch1 == 1)
# # weights = wei1.copy()
# # weights[where_should_weights_change] = np.random.normal(wei1, 1, (neuron_count, neuron_count))[where_should_weights_change]
# # # weights = np.multiply(weights, ch1)
# # weights = np.multiply(weights, link1)
# # # print(f"weights: \n {wei_moves}")
# # ch1b = np.zeros((1, neuron_count))
# # ch1b[np.random.random((1, neuron_count)) <= 0.75] = 1
# # ch1b[0, :input_size] = 0
# # where_should_bias_change = np.where(ch1b == 1)
# # biases = bia1.copy()
# # biases[where_should_bias_change] = np.random.normal(bia1, 1, (1, neuron_count))[where_should_bias_change]
# # ch_it = random.random()
# # if ch_it <= 0.75:
# #     print(f"change_it: \n{try_choose_different(maxit, [1, 2, 3, 4, 5])}")
# # ch_l = np.zeros(wei1.shape)
# # ch_l[np.random.random(wei1.shape) <= 0.75] = 1
# # ch_l[:, :input_size] = 0
# # ch_l[-output_size:, :] = 0
# # np.fill_diagonal(ch_l, 0)
# # # print(f"ch_l: \n{ch_l}")
# # links = link1.copy()
# # links[ch_l == 1] = 1 - links[ch_l == 1]
# # # print(f"lc: \n{link2}")
# # edge_status = link1 - links
# # # print(f"status_edges: \n{edge_status}")
# # minW = np.min(weights)
# # maxW = np.max(weights)
# # new_edges_weights = weights.copy()
# # added_edges = np.where(edge_status == -1)
# # new_edges_weights[added_edges] = np.random.uniform(minW, maxW, (neuron_count, neuron_count))[added_edges]
# # new_edges_weights = np.multiply(new_edges_weights, links)
# # weights = new_edges_weights
# # ca_1 = random.random()
# # if ca_1 <= 0.75:
# #     print(f"ca_1: {try_choose_different(ReLu(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
# # ca_2 = random.random()
# # if ca_2 <= 0.75:
# #     print(f"ca_2: {try_choose_different(ReLu(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
# # aggc = random.random()
# # if aggc <= 0.75:
# #     print(f"aggr: {try_choose_different(TanH(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
# # hc_c = random.random()
# # if hc_c <= 0.75:
# #     new_hc = try_choose_different(2, [0, 1, 2, 3, 4])
# #     print(f"new hc: {new_hc}")
# #     if new_hc < hidden_size:
# #         to_preserve = [0, 1, 2, 3, 4]
# #         brr = choose_without_repetition(list(range(input_size, input_size + hidden_size)), hidden_size - new_hc)
# #         for i in brr:
# #             to_preserve.remove(i)
# #         to_preserve = np.array(to_preserve).reshape(1, -1)
# #
# #         links = links[to_preserve[0, :, None], to_preserve]
# #         weights = weights[to_preserve[0, :, None], to_preserve]
# #         biases = biases[0, to_preserve]
# #         new_af = []
# #         print(f"to_preserve: {to_preserve}")
# #
# #     elif new_hc > hidden_size:
# #         act_set = [ReLu(), Sigmoid(), GaussAct(), TanH()]
# #         link3 = get_links(input_size, output_size, input_size + output_size + new_hc)
# #         link3[:input_size + hidden_size, :input_size + hidden_size] = links[:input_size + hidden_size, :input_size + hidden_size]
# #         link3[:input_size + hidden_size, -output_size:] = links[:input_size + hidden_size, -output_size:]
# #
# #         links = link3
# #
# #         minW = np.min(weights)
# #         maxW = np.max(weights)
# #         weight3 = np.random.uniform(minW, maxW, (1 + 2 + new_hc, 1 + 2 + new_hc))
# #         weight3[:input_size + hidden_size, :input_size + hidden_size] = weights[:input_size + hidden_size, :input_size + hidden_size]
# #         weight3[:input_size + hidden_size, -output_size:] = weights[:input_size + hidden_size, -output_size:]
# #         weight3 = np.multiply(weight3, link3)
# #
# #         weights = weight3
# #
# #         minB = np.min(biases)
# #         maxB = np.max(biases)
# #         bia3 = np.random.uniform(minB, maxB, (1, input_size + output_size + new_hc))
# #         bia3[0, :input_size + hidden_size] = biases[0, :input_size + hidden_size]
# #         bia3[0, -output_size:] = biases[0, -output_size:]
# #
# #         biases = bia3
# #
# #         for i in range(new_hc - hidden_size):
# #             print(f"inc_add_{i+1}: {act_set[random.randint(0, len(act_set) - 1)]}")
# #
# #
# # print(f"new_links: \n {links}")
# # print(f"new_weights: \n{weights}")
# # print(f"biases: \n {biases}")
# # mut_rad_change = random.random()
# # if mut_rad_change <= 0.75:
# #     print(f"mut_rad: \n {random.uniform(0, 1)}")
# # wb_prob_change = random.random()
# # if wb_prob_change <= 0.75:
# #     print(f"wb_prob: \n {random.uniform(0.05, 0.1)}")
# # s_prob_change = random.random()
# # if s_prob_change <= 0.75:
# #     print(f"s_prob: \n {random.uniform(0.6, 0.7)}")
# # p_prob_change = random.random()
# # if p_prob_change <= 0.75:
# #     print(f"p_prob: \n {random.uniform(0.8, 1)}")
# # c_prob_change = random.random()
# # if c_prob_change <= 0.75:
# #     print(f"c_prob: \n {random.uniform(0.22, 0.33)}")
# # r_prob_change = random.random()
# # if r_prob_change <= 0.75:
# #     print(f"r_prob: \n {random.uniform(0.44, 0.55)}")
# # test_struct_mutation()
# # test_struct_mutation_2()
# # test_struct_mutation_3()
#
#
