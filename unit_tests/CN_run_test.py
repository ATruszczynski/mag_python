import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet

def test_cn_run():
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
    actFuns = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax())

    result = net.run(np.array([[0], [0]]))

    assert np.all(np.isclose(net.inp, np.array([[0, 0, 0.5, 0.5, 0.744918662, 0.122459331, 0.461494578]]), atol=1e-5))
    assert np.all(np.isclose(net.act, np.array([[0, 0, 0.622459331, 0.622459331, 0.678070495, 0.416043846, 0.583956154]]), atol=1e-5))
    assert np.all(np.isclose(result, np.array([[0.416043846, 0.583956154]]), atol=1e-5))

# test_cn_run()
