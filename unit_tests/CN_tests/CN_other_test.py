import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet


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
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax())


    net.run(np.array([[0], [1]]))
    net2 = net.copy()


    assert np.array_equal(net.links, np.array([[0, 0, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(net.weights, np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                 [0, 0, 0, -1, 0.5, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 2, 0, 1],
                                                 [0, 0, 0, 0, 0, 0, 0.5],
                                                 [0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(net.bias, np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]))
    assert len(net.actFuns) == 7
    assert net.actFuns[0] is None
    assert net.actFuns[1] is None
    assert net.actFuns[2].to_string() == Sigmoid().to_string()
    assert net.actFuns[3].to_string() == TanH().to_string()
    assert net.actFuns[4].to_string() == ReLu().to_string()
    assert net.actFuns[5] is None
    assert net.actFuns[6] is None
    assert net.aggrFun.to_string() == Softmax().to_string()
    assert len(net.hidden_comp_order) == 3
    assert net.hidden_comp_order[0] == 2
    assert net.hidden_comp_order[1] == 3
    assert net.hidden_comp_order[2] == 4

    assert np.array_equal(net2.links, np.array([[0, 0, 1, 0, 0, 0, 0],
                                               [0, 0, 0, 1, 1, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0],
                                               [0, 0, 0, 0, 1, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(net2.weights, np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                 [0, 0, 0, -1, 0.5, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 2, 0, 1],
                                                 [0, 0, 0, 0, 0, 0, 0.5],
                                                 [0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(net2.bias, np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]))
    assert np.array_equal(net2.inp, np.array([[0, 0, 0, 0, 0, 0, 0]]))
    assert np.array_equal(net2.act, np.array([[0, 0, 0, 0, 0, 0, 0]]))
    assert len(net2.actFuns) == 7
    assert net2.actFuns[0] is None
    assert net2.actFuns[1] is None
    assert net2.actFuns[2].to_string() == Sigmoid().to_string()
    assert net2.actFuns[3].to_string() == TanH().to_string()
    assert net2.actFuns[4].to_string() == ReLu().to_string()
    assert net2.actFuns[5] is None
    assert net2.actFuns[6] is None
    assert net2.aggrFun.to_string() == Softmax().to_string()
    assert net2.hidden_comp_order is None

