# import numpy as np
import random
from math import log10

from ann_point.Functions import *
# from ann_point.AnnPoint2 import *
# from utility.Mut_Utility import resize_layer
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.LsmCrossoverOperator import LsmCrossoverOperator, find_possible_cuts, get_link_weights_biases_acts
from neural_network.LsmNetwork import LsmNetwork
from utility.TestingUtility import assert_chaos_network_properties

#TODO - B - test multiple runs vs single run (done?)
#TODO - B - ec test multiple runs vs single run? (done?)
from utility.Utility import compare_lists, choose_without_repetition
from utility.Utility2 import assert_acts_same


def test_find_cuts():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 20), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 p_rad=(0, 0)) # values irrelevant aside from neuron count

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0 , 0, 3, 0, 5],
                     [0 , 7, 0, 0, 6],
                     [0 , 0, 0, 0, 0],
                     [0 , 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None]

    link2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0,  0,  0],
                     [0, 0, 10, 20, 0],
                     [0, 0, 0,  30, 40],
                     [0, 0, 0,  0,  0],
                     [0, 0, 0,  0,  0]])
    bia2 = np.array([[0., -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=-2,
                     multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20,
                     multi=-30, p_prob=-40, c_prob=-50, p_rad=-60)

    cuts = find_possible_cuts(cn1, cn2, hrange)

    assert len(cuts) == 9
    assert compare_lists(cuts[0], [1, 0, 1, 2])
    assert compare_lists(cuts[1], [1, 0, 2, 1])
    assert compare_lists(cuts[2], [1, 0, 3, 0])
    assert compare_lists(cuts[3], [2, 1, 1, 2])
    assert compare_lists(cuts[4], [2, 1, 2, 1])
    assert compare_lists(cuts[5], [2, 1, 3, 0])
    assert compare_lists(cuts[6], [3, 2, 1, 2])
    assert compare_lists(cuts[7], [3, 2, 2, 1])
    assert compare_lists(cuts[8], [3, 2, 3, 0])

def test_find_cuts_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 20), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 p_rad=(0, 0)) # values irrelevant aside from neuron count

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0 , 0, 3, 0, 5],
                     [0 , 7, 0, 0, 6],
                     [0 , 0, 0, 0, 0],
                     [0 , 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None]

    link2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0,  0,  0],
                     [0, 0, 10, 20, 0],
                     [0, 0, 0,  30, 40],
                     [0, 0, 0,  0,  0],
                     [0, 0, 0,  0,  0]])
    bia2 = np.array([[0., -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=-2,
                     multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20,
                     multi=-30, p_prob=-40, c_prob=-50, p_rad=-60)

    cuts = find_possible_cuts(cn1, cn2, hrange)

    assert len(cuts) == 7
    assert compare_lists(cuts[0], [1, 0, 1, 2])
    assert compare_lists(cuts[1], [1, 0, 2, 1])
    assert compare_lists(cuts[2], [2, 1, 1, 2])
    assert compare_lists(cuts[3], [2, 1, 2, 1])
    assert compare_lists(cuts[4], [2, 1, 3, 0])
    assert compare_lists(cuts[5], [3, 2, 2, 1])
    assert compare_lists(cuts[6], [3, 2, 3, 0])

def test_find_cuts_3():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 20), (0, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 p_rad=(0, 0)) # values irrelevant aside from neuron count

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0 , 0, 3, 0, 5],
                     [0 , 7, 0, 0, 6],
                     [0 , 0, 0, 0, 0],
                     [0 , 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), ReLu(), None]

    link2 = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 1, 0],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    bia2 = np.array([[0., -20, -30, -40, -50, -60]])
    actFuns2 = [None, TanH(), TanH(), TanH(), TanH(), None]

    cn1 = LsmNetwork(input_size=1, output_size=1, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=-2,
                     multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=1, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20,
                     multi=-30, p_prob=-40, c_prob=-50, p_rad=-60)

    cuts = find_possible_cuts(cn1, cn2, hrange)

    assert len(cuts) == 18

    assert compare_lists(cuts[0], [1, 0, 1, 4])
    assert compare_lists(cuts[1], [1, 0, 2, 3])
    assert compare_lists(cuts[2], [1, 0, 3, 2])
    assert compare_lists(cuts[3], [1, 0, 4, 1])

    assert compare_lists(cuts[4], [2, 1, 1, 4])
    assert compare_lists(cuts[5], [2, 1, 2, 3])
    assert compare_lists(cuts[6], [2, 1, 3, 2])
    assert compare_lists(cuts[7], [2, 1, 4, 1])
    assert compare_lists(cuts[8], [2, 1, 5, 0])

    assert compare_lists(cuts[9], [3, 2, 1, 4])
    assert compare_lists(cuts[10], [3, 2, 2, 3])
    assert compare_lists(cuts[11], [3, 2, 3, 2])
    assert compare_lists(cuts[12], [3, 2, 4, 1])
    assert compare_lists(cuts[13], [3, 2, 5, 0])

    assert compare_lists(cuts[14], [4, 3, 2, 3])
    assert compare_lists(cuts[15], [4, 3, 3, 2])
    assert compare_lists(cuts[16], [4, 3, 4, 1])
    assert compare_lists(cuts[17], [4, 3, 5, 0])

def test_simple_crossover():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 20), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 p_rad=(0, 0)) # values irrelevant aside from neuron count

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0 , 0, 3, 0, 5],
                     [0 , 7, 0, 0, 6],
                     [0 , 0, 0, 0, 0],
                     [0 , 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None]

    link2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0,  0,  0],
                     [0, 0, 10, 20, 0],
                     [0, 0, 0,  30, 40],
                     [0, 0, 0,  0,  0],
                     [0, 0, 0,  0,  0]])
    bia2 = np.array([[0., -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=log10(0.5),
                     multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20,
                     multi=-30, p_prob=-40, c_prob=-50, p_rad=-60)
    co = LsmCrossoverOperator(hrange)

    seed = 1006
    random.seed(seed)
    np.random.seed(seed)

    cn3, cn4 = co.crossover(cn1, cn2)

    ##################################################################

    # assert_chaos_network_properties(net=cn1,
    #                                 desired_input_size=1,
    #                                 desired_output_size=2,
    #                                 desired_neuron_count=5,
    #                                 desired_hidden_start_index=1,
    #                                 desired_hidden_end_index=3,
    #                                 desired_hidden_count=2,
    #                                 desired_links=np.array([[0, 1, 1, 0, 0],
    #                                                         [0, 0, 1, 0, 1],
    #                                                         [0, 1, 0, 0, 1],
    #                                                         [0, 0, 0, 0, 0],
    #                                                         [0, 0, 0, 0, 0]]),
    #                                 desired_weights=np.array([[0., 1, 2, 0, 0],
    #                                                           [0 , 0, 3, 0, 5],
    #                                                           [0 , 7, 0, 0, 6],
    #                                                           [0 , 0, 0, 0, 0],
    #                                                           [0 , 0, 0, 0, 0]]),
    #                                 desired_biases=np.array([[0., -2, -3, -4, -5]]),
    #                                 desired_actFun=[None, ReLu(), ReLu(), None, None],
    #                                 desired_aggr=SincAct(),
    #                                 desired_maxit=1,
    #                                 desired_mut_rad=-1,
    #                                 desired_wb_prob=-2,
    #                                 desired_s_prob=-3,
    #                                 desired_p_prob=log10(0.4),
    #                                 desired_c_prob=-5,
    #                                 desired_r_prob=-6)
    #
    # ##################################################################
    #
    # assert_chaos_network_properties(net=cn2,
    #                                 desired_input_size=1,
    #                                 desired_output_size=2,
    #                                 desired_neuron_count=5,
    #                                 desired_hidden_start_index=1,
    #                                 desired_hidden_end_index=3,
    #                                 desired_hidden_count=2,
    #                                 desired_links=np.array([[0, 0, 0, 0, 0],
    #                                                         [0, 0, 1, 1, 0],
    #                                                         [0, 0, 0, 1, 1],
    #                                                         [0, 0, 0, 0, 0],
    #                                                         [0, 0, 0, 0, 0]]),
    #                                 desired_weights=np.array([[0, 0, 0,  0,  0],
    #                                                           [0, 0, 10, 20, 0],
    #                                                           [0, 0, 0,  30, 40],
    #                                                           [0, 0, 0,  0,  0],
    #                                                           [0, 0, 0,  0,  0]]),
    #                                 desired_biases=np.array([[0, -20, -30, -40, -50]]),
    #                                 desired_actFun=[None, TanH(), TanH(), None, None],
    #                                 desired_aggr=GaussAct(),
    #                                 desired_maxit=10,
    #                                 desired_mut_rad=-10,
    #                                 desired_wb_prob=-20,
    #                                 desired_s_prob=-30,
    #                                 desired_p_prob=-40,
    #                                 desired_c_prob=-50,
    #                                 desired_r_prob=-60)

    ##################################################################

    assert_chaos_network_properties(net=cn3,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=4,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=2,
                                    desired_hidden_count=1,
                                    desired_links=np.array([[0, 1, 0, 0],
                                                            [0, 0, 0, 1],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 1, 0, 0],
                                                              [0, 0, 0, 5],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, -2, -4, -5]]),
                                    desired_actFun=[None, ReLu(), None, None],
                                    desired_aggr=GaussAct(),
                                    desired_maxit=10,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=log10(0.5),
                                    desired_s_prob=-30,
                                    desired_p_prob=-40,
                                    desired_c_prob=-50,
                                    desired_r_prob=-6)

    ##################################################################

    assert_chaos_network_properties(net=cn4,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0, 0, 0, 0, 0],
                                                            [0, 0, 1, 1, 0],
                                                            [0, 0, 0, 1, 1],
                                                            [0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0,  0, 0],
                                                              [0, 0, 10, 20, 0],
                                                              [0, 0, 0,  30, 40],
                                                              [0, 0, 0,  0, 0],
                                                              [0, 0, 0,  0, 0]]),
                                    desired_biases=np.array([[0., -20, -30, -40, -5]]),
                                    desired_actFun=[None, TanH(), TanH(), None, None],
                                    desired_aggr=SincAct(),
                                    desired_maxit=1,
                                    desired_mut_rad=-10,
                                    desired_wb_prob=-20,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-60)

def test_simple_crossover_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 p_rad=(0, 0))

    link1 = np.array([[0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1],
                      [0, 1, 0, 1, 1],
                      [0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0,  -1,  -2,  -3,   0],
                     [0,   0,  -4,  -5,  -6],
                     [0,  -7,   0,  -8,  -9],
                     [0, -10, -11,   0, -12],
                     [0,   0,   0,   0,   0]])
    bia1 = np.array([[0., -2, -3, -4, -5.]])
    actFuns1 = [None, ReLu(), ReLu(), ReLu(), None]

    link2 = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1],
                      [0, 1, 0, 1, 1, 1, 1, 1],
                      [0, 1, 1, 0, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0, 1, 1, 1],
                      [0, 1, 1, 1, 1, 0, 1, 1],
                      [0, 1, 1, 1, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    wei2 = np.array([[0,   1,   2,   3,   4,   5,   6,   0],
                     [0,   0,   7,   8,   9,  10,  11,  12],
                     [0,  13,   0,  14,  15,  16,  17,  18],
                     [0,  19,  20,   0,  21,  22,  23,  24],
                     [0,  25,  26,  27,   0,  28,  29,  30],
                     [0,  31,  32,  33,  34,   0,  35,  36],
                     [0,  37,  38,  39,  40,  41,   0,  42],
                     [0,   0,   0,   0,   0,   0,   0,   0]])
    bia2 = np.array([[0, -20, -30, -40, -50, -60, -70, -80]])
    actFuns2 = [None, TanH(), TanH(), TanH(), TanH(), TanH(), TanH(), None]

    cn1 = LsmNetwork(input_size=1, output_size=1, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=log10(0.4), multi=-3,
                     p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=1, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20, multi=-30,
                     p_prob=-40, c_prob=-50, p_rad=-60)

    co = LsmCrossoverOperator(hrange)

    seed = 1002
    random.seed(seed)
    np.random.seed(seed)
    cn3, cn4 = co.crossover(cn1, cn2)


    ##################################################################

    # assert_chaos_network_properties(net=cn1,
    #                                 desired_input_size=1,
    #                                 desired_output_size=1,
    #                                 desired_neuron_count=5,
    #                                 desired_hidden_start_index=1,
    #                                 desired_hidden_end_index=4,
    #                                 desired_hidden_count=3,
    #                                 desired_links=np.array([[0, 1, 1, 1, 0],
    #                                                         [0, 0, 1, 1, 1],
    #                                                         [0, 1, 0, 1, 1],
    #                                                         [0, 1, 1, 0, 1],
    #                                                         [0, 0, 0, 0, 0]]),
    #                                 desired_weights=np.array([[0,  -1,  -2,  -3,   0],
    #                                                           [0,   0,  -4,  -5,  -6],
    #                                                           [0,  -7,   0,  -8,  -9],
    #                                                           [0, -10, -11,   0, -12],
    #                                                           [0,   0,   0,   0,   0]]),
    #                                 desired_biases=np.array([[0., -2, -3, -4, -5.]]),
    #                                 desired_actFun=[None, ReLu(), ReLu(), ReLu(), None],
    #                                 desired_aggr=SincAct(),
    #                                 desired_maxit=1,
    #                                 desired_mut_rad=-1,
    #                                 desired_wb_prob=-2,
    #                                 desired_s_prob=-3,
    #                                 desired_p_prob=-4,
    #                                 desired_c_prob=-5,
    #                                 desired_r_prob=-6)
    #
    # ##################################################################
    #
    # assert_chaos_network_properties(net=cn2,
    #                                 desired_input_size=1,
    #                                 desired_output_size=1,
    #                                 desired_neuron_count=8,
    #                                 desired_hidden_start_index=1,
    #                                 desired_hidden_end_index=7,
    #                                 desired_hidden_count=6,
    #                                 desired_links=np.array([[0, 1, 1, 1, 1, 1, 1, 0],
    #                                                         [0, 0, 1, 1, 1, 1, 1, 1],
    #                                                         [0, 1, 0, 1, 1, 1, 1, 1],
    #                                                         [0, 1, 1, 0, 1, 1, 1, 1],
    #                                                         [0, 1, 1, 1, 0, 1, 1, 1],
    #                                                         [0, 1, 1, 1, 1, 0, 1, 1],
    #                                                         [0, 1, 1, 1, 1, 1, 0, 1],
    #                                                         [0, 0, 0, 0, 0, 0, 0, 0]]),
    #                                 desired_weights=np.array([[0,   1,   2,   3,   4,   5,   6,   0],
    #                                                           [0,   0,   7,   8,   8,  10,  11,  12],
    #                                                           [0,  13,   0,  14,  15,  16,  17,  18],
    #                                                           [0,  19,  20,   0,  21,  22,  23,  24],
    #                                                           [0,  25,  26,  27,   0,  28,  29,  30],
    #                                                           [0,  31,  32,  33,  34,   0,  35,  36],
    #                                                           [0,  37,  38,  39,  40,  41,   0,  42],
    #                                                           [0,   0,   0,   0,   0,   0,   0,   0]]),
    #                                 desired_biases=np.array([[0, -20, -30, -40, -50, -60, -70, -80]]),
    #                                 desired_actFun=[None, TanH(), TanH(), TanH(), TanH(), TanH(), TanH(), None],
    #                                 desired_aggr=GaussAct(),
    #                                 desired_maxit=10,
    #                                 desired_mut_rad=-10,
    #                                 desired_wb_prob=-20,
    #                                 desired_s_prob=-30,
    #                                 desired_p_prob=-40,
    #                                 desired_c_prob=-50,
    #                                 desired_r_prob=-60)


    ##################################################################

    assert_chaos_network_properties(net=cn3,
                                    desired_input_size=1,
                                    desired_output_size=1,
                                    desired_neuron_count=7,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=6,
                                    desired_hidden_count=5,
                                    desired_links=np.array([[0,   1,   1,   1,   1,   1,   0],
                                                            [0,   0,   1,   1,   1,   1,   1],
                                                            [0,   1,   0,   1,   1,   1,   1],
                                                            [0,   1,   1,   0,   1,   1,   1],
                                                            [0,   1,   1,   1,   0,   1,   1],
                                                            [0,   1,   1,   1,   1,   0,   1],
                                                            [0,   0,   0,   0,   0,   0,   0]]),
                                    desired_weights=np.array([[0,  -1,  -2,   4,   5,   6,   0],
                                                              [0,   0,  -4,  15,  16,  17,  -6],
                                                              [0,  -7,   0,  21,  22,  23,  -9],
                                                              [0, -10, -11,   0,  28,  29,  30],
                                                              [0,  32,  33,  34,   0,  35,  36],
                                                              [0,  38,  39,  40,  41,   0,  42],
                                                              [0,   0,   0,   0,   0,   0,   0]]),
                                    desired_biases=np.array([[0., -2, -3, -50, -60, -70, -5]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), TanH(), TanH(), None],
                                    desired_aggr=GaussAct(),
                                    desired_maxit=1,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=log10(0.4),
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-50,
                                    desired_r_prob=-60)

    ###################################################################

    assert_chaos_network_properties(net=cn4,
                                    desired_input_size=1,
                                    desired_output_size=1,
                                    desired_neuron_count=10,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=9,
                                    desired_hidden_count=8,
                                    desired_links=np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                                                            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                                            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                                                            [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                                                            [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                                                            [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
                                                            [0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
                                                            [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0,  -1,  -2,  -3,   2,   3,   4,   5,   6,   0],
                                                              [0,   0,  -4,  -5,   0,   0,   0,   0,   0,  -6],
                                                              [0,  -7,   0,  -8,   0,   0,   0,   0,   0,  -9],
                                                              [0, -10, -11,   0,   7,   8,   9,  10,  11, -12],
                                                              [0,   0,   0,  13,   0,  14,  15,  16,  17,  18],
                                                              [0,   0,   0,  19,  20,   0,  21,  22,  23,  24],
                                                              [0,   0,   0,  25,  26,  27,   0,  28,  29,  30],
                                                              [0,   0,   0,  31,  32,  33,  34,   0,  35,  36],
                                                              [0,   0,   0,  37,  38,  39,  40,  41,   0,  42],
                                                              [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -30, -40, -50, -60, -70, -5]]),
                                    desired_actFun=[None, ReLu(), ReLu(), ReLu(), TanH(), TanH(), TanH(), TanH(), TanH(), None],
                                    desired_aggr=SincAct(),
                                    desired_maxit=10,
                                    desired_mut_rad=-10,
                                    desired_wb_prob=-20,
                                    desired_s_prob=-30,
                                    desired_p_prob=-40,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6)

def test_pieceing_together():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    link1 = np.array([[0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1],
                      [0, 1, 0, 1, 1],
                      [0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0,  -1,  -2,  -3,   0],
                     [0,   0,  -4,  -5,  -6],
                     [0,  -7,   0,  -8,  -9],
                     [0, -10, -11,   0, -12],
                     [0,   0,   0,   0,   0]])
    bia1 = np.array([[0., -2, -3, -4, -5.]])
    actFuns1 = [None, ReLu(), ReLu(), ReLu(), None]

    link2 = np.array([[0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1],
                      [0, 1, 0, 1, 1, 1, 1, 1],
                      [0, 1, 1, 0, 1, 1, 1, 1],
                      [0, 1, 1, 1, 0, 1, 1, 1],
                      [0, 1, 1, 1, 1, 0, 1, 1],
                      [0, 1, 1, 1, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    wei2 = np.array([[0,   1,   2,   3,   4,   5,   6,   0],
                     [0,   0,   7,   8,   8,  10,  11,  12],
                     [0,  13,   0,  14,  15,  16,  17,  18],
                     [0,  19,  20,   0,  21,  22,  23,  24],
                     [0,  25,  26,  27,   0,  28,  29,  30],
                     [0,  31,  32,  33,  34,   0,  35,  36],
                     [0,  37,  38,  39,  40,  41,   0,  42],
                     [0,   0,   0,   0,   0,   0,   0,   0]])
    bia2 = np.array([[0, -20, -30, -40, -50, -60, -70, -80]])
    actFuns2 = [None, TanH(), TanH(), TanH(), TanH(), TanH(), TanH(), None]

    cn1 = LsmNetwork(input_size=1, output_size=1, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                     aggrFun=SincAct(), net_it=1, mutation_radius=-1, swap_prob=-2, multi=-3,
                     p_prob=-4, c_prob=-5, p_rad=-6)
    cn2 = LsmNetwork(input_size=1, output_size=1, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                     aggrFun=GaussAct(), net_it=10, mutation_radius=-10, swap_prob=-20, multi=-30,
                     p_prob=-40, c_prob=-50, p_rad=-60)

    l, w, b, a = get_link_weights_biases_acts(cn1, cn2, [3, 2, 2, 5])

    assert np.array_equal(l, np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0],
                                       [0, 0, 1, 1, 0, 0, 0, 0, 1],
                                       [0, 1, 0, 1, 1, 1, 1, 1, 1],
                                       [0, 1, 1, 0, 1, 1, 1, 1, 1],
                                       [0, 0, 1, 1, 0, 1, 1, 1, 1],
                                       [0, 0, 1, 1, 1, 0, 1, 1, 1],
                                       [0, 0, 1, 1, 1, 1, 0, 1, 1],
                                       [0, 0, 1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0]]))

    assert np.array_equal(w, np.array([[0,  -1,  -2,   2,   3,   4,   5,   6,   0],
                                       [0,   0,  -4,  -5,   0,   0,   0,   0,  -6],
                                       [0,  -7,   0,   7,   8,   8,  10,  11,  -9],
                                       [0, -10, -11,   0,  14,  15,  16,  17,  18],
                                       [0,   0,  19,  20,   0,  21,  22,  23,  24],
                                       [0,   0,  25,  26,  27,   0,  28,  29,  30],
                                       [0,   0,  31,  32,  33,  34,   0,  35,  36],
                                       [0,   0,  37,  38,  39,  40,  41,   0,  42],
                                       [0,   0,   0,   0,   0,   0,   0,   0,   0]]))

    assert np.array_equal(b, np.array([[0., -2, -3, -30, -40, -50, -60, -70, -80]]))
    assert_acts_same(a, [None, ReLu(), ReLu(), TanH(), TanH(), TanH(), TanH(), TanH(), None])


