import numpy as np
import pytest

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.TestingUtility import compare_chaos_network

def test_cn_test_cm_only():
    weights = np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , -2, 1  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 2   , 0  ],
                        [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                        [0, 0, -2 , -1, 0  , -0.5, 0.5],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ]])

    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4, c_prob=-5, dstr_mut_prob=-6)


    inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
              np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]

    outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
               np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]

    test_res = net.test(test_input=inputs, test_output=outputs)

    assert len(test_res) == 1
    assert np.array_equal(test_res[0], np.array([[3, 0],
                                                 [2, 1]]))

def test_cn_test_full():
    weights = np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , -2, 1  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 2   , 0  ],
                        [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                        [0, 0, -2 , -1, 0  , -0.5, 0.5],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ]])

    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4, c_prob=-5, dstr_mut_prob=-6)


    inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
              np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]

    outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
              np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]

    test_res = net.test(test_input=inputs, test_output=outputs, lf=QuadDiff())

    assert len(test_res) == 2
    assert np.array_equal(test_res[0], np.array([[3, 0],
                                                 [2, 1]]))

    assert test_res[1] == pytest.approx(1.38408458, abs=1e-5)

def test_cn_test_full_2():
    weights = np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                        [0, 0, -2  , -2, 1  , 0   , 0  ],
                        [0, 0, 0  , 1 , 0  , 1   , 0.9],
                        [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                        [0, 0, -2 , -1, 0  , -0.5, 0.5],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ]])

    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1],
                        [0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 1.5, 0.5, -2.5, -0.55, -0.5]])
    actFuns = [None, None, ReLu(), TanH(), Identity(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=TanH(),
                   net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4, c_prob=-5, dstr_mut_prob=-6)


    inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
              np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]

    outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
               np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]

    test_res = net.test(test_input=inputs, test_output=outputs, lf=ChebyshevLoss())

    assert len(test_res) == 2
    assert np.array_equal(test_res[0], np.array([[3, 0],
                                                 [3, 0]]))

    assert test_res[1] == pytest.approx(5.921408317, abs=1e-5)


#
# weights = np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
#                     [0, 0, -2  , -2, 1  , 0   , 0  ],
#                     [0, 0, 0  , 1 , 0  , 1   , 0.9],
#                     [0, 0, 1  , 0 , 0.5, 0   , 0.5],
#                     [0, 0, -2 , -1, 0  , -0.5, 0.5],
#                     [0, 0, 0  , 0 , 0  , 0   , 0  ],
#                     [0, 0, 0  , 0 , 0  , 0   , 0  ]])
#
# links =   np.array([[0, 0, 1, 0, 0, 0, 0],
#                     [0, 0, 1, 1, 1, 0, 0],
#                     [0, 0, 0, 1, 0, 1, 1],
#                     [0, 0, 1, 0, 1, 0, 1],
#                     [0, 0, 1, 1, 0, 1, 1],
#                     [0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])
# bias = np.array([[0, 0, 1.5, 0.5, -2.5, -0.55, -0.5]])
# actFuns = [None, None, ReLu(), TanH(), Identity(), None, None]
# net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=TanH(),
#                net_it=2, mutation_radius=-1, sqr_mut_prob=-2, lin_mut_prob=-3, p_mutation_prob=-4, c_prob=-5, dstr_mut_prob=-6)
#
# input = np.array([[0, 0, 1, 1, -1, -1],
#                   [0, 1, 0, 1,  1, -1]])
#
# output = np.array([[1, 0, 0, 1, 0, 1],
#                    [0, 1, 1, 0, 1, 0]])
#
# inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
#           np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]
#
# outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
#            np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]
#
# result = net.run(input)
# print(result)
# ef = ChebyshevLoss()
#
# sum = 0
# for i in range(len(inputs)):
#     sum += ef.compute(result[:, i].reshape(-1, 1), outputs[i])
#     print(sum)
#
# test_cn_test_full_2()