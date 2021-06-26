import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.TestingUtility import compare_chaos_network


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
                   maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)


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
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
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
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3)

