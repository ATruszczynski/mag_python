import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet, efficiency
from utility.TestingUtility import compare_chaos_network
import random

from utility.Utility import generate_counting_problem
from utility.Utility2 import get_weight_mask

def test_CN_constr():
    weights = np.array([[0, 0, 0.5, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 2, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0.5],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), TanH(), ReLu(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4,
                   c_prob=-5, dstr_mut_prob=-6)

    compare_chaos_network(net=net,
                          desired_input_size=2,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=5,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                          desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                          desired_aggr=Softmax(),
                          desired_maxit=2,
                          desired_mut_rad=-1,
                          desired_wb_prob=-2,
                          desired_s_prob=-3,
                          desired_p_prob=-4,
                          desired_c_prob=-5,
                          desired_r_prob=-6,
                          desired_hidden_comp_order=None,
                          desired_inp=np.zeros((0,0)),
                          desired_act=np.zeros((0,0)))


def test_CN_copy():
    weights = np.array([[0, 0, 0.5, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 2, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0.5],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), TanH(), ReLu(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4,
                   c_prob=-5, dstr_mut_prob=-6)


    net.run(np.array([[0], [1]]))
    net2 = net.copy()

    compare_chaos_network(net=net,
                          desired_input_size=2,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=5,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                          desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                          desired_aggr=Softmax(),
                          desired_maxit=2,
                          desired_mut_rad=-1,
                          desired_wb_prob=-2,
                          desired_s_prob=-3,
                          desired_p_prob=-4,
                          desired_c_prob=-5,
                          desired_r_prob=-6,
                          desired_hidden_comp_order=[2, 3, 4],
                          desired_inp=np.array([[0, 0, 0.5, -0.5, -0.924234315, 0.122459331, -0.962117157]]).reshape(-1, 1),
                          desired_act=np.array([[0, 1, 0.622459331, -0.462117157, 0, 0.747359064, 0.252640936]]).reshape(-1, 1))



    compare_chaos_network(net=net2,
                          desired_input_size=2,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=5,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                          desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                          desired_aggr=Softmax(),
                          desired_maxit=2,
                          desired_mut_rad=-1,
                          desired_wb_prob=-2,
                          desired_s_prob=-3,
                          desired_p_prob=-4,
                          desired_c_prob=-5,
                          desired_r_prob=-6)

# TODO - A - make some similar test
# def test_cn_test_5():
#     seed = 1001
#     random.seed(seed)
#     np.random.seed(seed)
#
#     fives_i, fives_o = generate_counting_problem(1000, 5)
#
#     links = np.zeros((11, 11))
#     links[:5, 5:] = 1
#     # print(links)
#
#     wei = np.array([[-10.04604658, -9.74799651, -9.86356132, -9.65934997, -10.00416797],
#                     [-4.22234218,  -4.20543259, -4.03250108, -4.15155717, -4.30413714],
#                     [-0.38323539,  -0.26812115, -0.20560072, -0.30525856, -0.58350255],
#                     [2.01047055,   2.39063375, 2.32230205, 2.40134387, 2.25093187],
#                     [4.61636766,   4.41484224, 4.5685952, 4.60406886, 4.28679621],
#                     [7.32939472,   7.52108442, 7.63706206, 7.50769173, 7.39856456]])
#
#     weights = np.zeros((11, 11))
#     weights[:5, 5:] = wei.T
#
#     links = np.multiply(links, get_weight_mask(5, 6, 11))
#     weights = np.multiply(weights, get_weight_mask(5, 6, 11))
#
#     biases = np.zeros((1, 11))
#     biases[0, 5:] = np.array([[14.18236634 ],
#                               [11.33332699 ],
#                               [5.72165648  ],
#                               [-0.84340657 ],
#                               [-8.69929589 ],
#                               [-21.69464734]]).T
#
#     weights = weights / 100
#     biases = biases / 100
#
#     # print(weights)
#     # print(biases)
#
#     net = ChaosNet(input_size=5, output_size=6, links=links, weights=weights, biases=biases, actFuns=11 * [None],
#                    aggrFun=Softmax(), net_it=1, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3,
#                    p_mutation_prob=-4, c_prob=-5, dstr_mut_prob=-6)
#     test_res = net.test(fives_i, fives_o, QuadDiff())
#
#     assert efficiency(test_res[0]) == 1.0



# test_cn_test_5()
