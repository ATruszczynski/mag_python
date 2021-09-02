import numpy as np
import pytest

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.TestingUtility import assert_chaos_network_properties

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
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)


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
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)


    inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
              np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]

    outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
              np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]

    test_res = net.test(test_input=inputs, test_output=outputs, lf=QuadDiff())

    assert len(test_res) == 3
    assert np.array_equal(test_res[0], np.array([[3, 0],
                                                 [2, 1]]))

    assert test_res[1] == pytest.approx(0.2306807643, abs=1e-5)
    assert test_res[2] == pytest.approx(0.4312329379, abs=1e-5)

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
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)


    inputs = [np.array([[0],[0]]), np.array([[0],[1]]), np.array([[1],[0]]),
              np.array([[1],[1]]), np.array([[-1],[1]]), np.array([[-1],[-1]])]

    outputs = [np.array([[1],[0]]), np.array([[0],[1]]), np.array([[0],[1]]),
               np.array([[1],[0]]), np.array([[0],[1]]), np.array([[1],[0]])]

    test_res = net.test(test_input=inputs, test_output=outputs, lf=ChebyshevLoss())

    assert len(test_res) == 3
    assert np.array_equal(test_res[0], np.array([[3, 0],
                                                 [3, 0]]))

    assert test_res[1] == pytest.approx(0.98690138, abs=1e-5)
    assert test_res[2] == pytest.approx(0.999999388, abs=1e-5)




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
#                net_it=2, mutation_radius=-1, depr=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
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
# maxx = -1
# for i in range(len(inputs)):
#     val = ef.compute(result[:, i].reshape(-1, 1), outputs[i])
#     sum += val
#     maxx = max(val, maxx)
#
# print(sum/6)
# print(maxx)
#
# test_cn_test_cm_only()
# test_cn_test_full()
# test_cn_test_full_2()