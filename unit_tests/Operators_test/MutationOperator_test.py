import random
from math import log10

import numpy as np
# import pytest

# from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.LsmMutationOperator import *
from utility.Mut_Utility import *
from utility.TestingUtility import assert_chaos_network_properties

def test_struct_mutation():
    seed = 20021
    random.seed(seed)
    np.random.seed(seed)

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(1), log10(2)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.79), log10(0.81)))

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0, 0, 3, 0, 5],
                     [0, 7, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                     actFuns=actFuns1, aggrFun=TanH(), net_it=2, mutation_radius=log10(0.5), swap_prob=log10(0.5),
                     multi=log10(1.5), p_prob=log10(0.7), c_prob=log10(0.3), p_rad=log10(0.8))

    mo = LsmMutationOperator(hrange)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0, 1, 1, 0, 0],
                                                            [0, 0, 1, 0, 1],
                                                            [0, 1, 0, 0, 1],
                                                            [0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0., 1, 2, 0, 0],
                                                            [0, 0, 3, 0, 5],
                                                            [0, 7, 0, 0, 6],
                                                            [0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5]]),
                                    desired_actFun=[None, ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.5),
                                    desired_wb_prob=log10(0.5),
                                    desired_s_prob=log10(1.5),
                                    desired_p_prob=log10(0.7),
                                    desired_c_prob=log10(0.3),
                                    desired_r_prob=log10(0.8))


    # wb_pm = 0.5
    # s_pm = 0.8
    # p_pm = 0.7
    # r_pm = 0.4
    # c_pm = 0.3

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0., 0., 1., 0., 0.],
                                                            [0., 0., 1., 0., 1.],
                                                            [0., 1., 0., 0., 1.],
                                                            [0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0, 2, 0, 0],
                                                              [0, 0, 3, 0, 5],
                                                              [0, 6.97400427, 0, 0, 6],
                                                              [0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, -2, -3.19269902, -4, -4.59589825]]),
                                    desired_actFun=[None, ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-0.5305919,
                                    desired_wb_prob=-0.3566040,
                                    desired_s_prob=0.17609125,
                                    desired_p_prob=-0.12493873,
                                    desired_c_prob=-0.52287874,
                                    desired_r_prob=-0.091514981)

def test_struct_mutation_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(0.65), log10(100)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.59), log10(0.61)))

    link1 = np.array([[0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 2, 3, 4, 0, 0],
                     [0, 0, 5, 6, 0, 7],
                     [0, 8, 0, 0, 0, 9],
                     [0, 2, 0, 0, 3, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
                     aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), swap_prob=log10(0.75),
                     multi=log10(90), p_prob=log10(0.74),
                     c_prob=log10(0.15), p_rad=log10(0.595))

    mo = LsmMutationOperator(hrange)


    random.seed(20021234)
    np.random.seed(20021234)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 0, 1],
                                                  [0, 1, 0, 0, 0, 1],
                                                  [0, 1, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 2, 3, 4, 0, 0],
                                                    [0, 0, 5, 6, 0, 7],
                                                    [0, 8, 0, 0, 0, 9],
                                                    [0, 2, 0, 0, 3, 5],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.35),
                                    desired_wb_prob=log10(0.75),
                                    desired_s_prob=log10(90),
                                    desired_p_prob=log10(0.74),
                                    desired_c_prob=log10(0.15),
                                    desired_r_prob=log10(0.595))

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 1., 0.],
                                                            [0., 0., 0., 1., 1., 0.],
                                                            [0., 0., 1., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 1.72372606, 0.],
                                                              [0., 0., 0., 7.49100203, 5.29989391, 0.],
                                                              [0., 0., 3.70193457, 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.]]),
                                    desired_biases=np.array([[-0.        , -1.70775246, -3.09250415, -3.53797184 ,-5.79764051 ,-5.8394995 ]]),
                                    desired_actFun=[None, GaussAct(), TanH(), GaussAct(), None, None],
                                    desired_aggr=Sigmoid(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.405688206,
                                    desired_wb_prob=-0.09691001,
                                    desired_s_prob=2.0,
                                    desired_p_prob=-0.126183892,
                                    desired_c_prob=-1.0,
                                    desired_r_prob=-0.22302233)

def test_struct_mutation_3():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(0.65), log10(1000)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.59), log10(0.61)))

    link1 = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [0, 1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 2, 0, 0, 0, 0],
                     [0, 0, 5, 6, 0, 7],
                     [0, 8, 0, 0, 0, 9],
                     [0, 2, -1, 0, 3, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
                     aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), swap_prob=log10(0.75),
                     multi=log10(99), p_prob=log10(0.74),
                     c_prob=log10(0.15), p_rad=log10(0.595))

    mo = LsmMutationOperator(hrange)

    random.seed(1112111)
    np.random.seed(1112111)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 0, 0, 0, 0],
                                                            [0, 0, 1, 1, 0, 1],
                                                            [0, 1, 0, 0, 0, 1],
                                                            [0, 1, 1, 0, 1, 1],
                                                            [0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 2, 0, 0, 0, 0],
                                                              [0, 0, 5, 6, 0, 7],
                                                              [0, 8, 0, 0, 0, 9],
                                                              [0, 2, -1, 0, 3, 5],
                                                              [0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0.]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.35),
                                    desired_wb_prob=log10(0.75),
                                    desired_s_prob=log10(99),
                                    desired_p_prob=log10(0.74),
                                    desired_c_prob=log10(0.15),
                                    desired_r_prob=log10(0.595))

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0., 0., 1., 1., 0., 0.],
                                                            [0., 0., 0., 0., 1., 0.],
                                                            [0., 0., 0., 1., 1., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0., 5.03743713, -0.01032964, 0., 0.],
                                                              [0., 0., 0., 0., 4.4777615, 0.],
                                                              [0., 0., 0., 2.52434007, 5.43532301, 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.]]),
                                    desired_biases=np.array([[-0.        , -2.63499526, -3.3678456 , -3.79587563, -5.10905153, -6.25300831]]),
                                    desired_actFun=[None, TanH(), Sigmoid(), GaussAct(), None, None],
                                    desired_aggr=Sigmoid(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.234027442,
                                    desired_wb_prob=-0.1249387366,
                                    desired_s_prob=3.0,
                                    desired_p_prob=-0.12493873,
                                    desired_c_prob=-0.8201796,
                                    desired_r_prob=-0.22617377)




