import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.Utility import *


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

    #TODO różne pola w inpucie
    result = net.run(np.array([[0], [0]]))

    assert np.all(np.isclose(net.inp, np.array([[0, 0, 0.5, 0.5, 0.744918662, 0.122459331, 0.461494578]]), atol=1e-5))
    assert np.all(np.isclose(net.act, np.array([[0, 0, 0.622459331, 0.622459331, 0.678070495, 0.416043846, 0.583956154]]), atol=1e-5))
    assert np.all(np.isclose(result, np.array([[0.416043846, 0.583956154]]), atol=1e-5))

    net2 = net.copy()
    result2 = net2.run(np.array([[0], [0]]))
    assert np.all(np.isclose(net2.inp, np.array([[0, 0, 0.5, 0.5, 0.744918662, 0.122459331, 0.461494578]]), atol=1e-5))
    assert np.all(np.isclose(net2.act, np.array([[0, 0, 0.622459331, 0.622459331, 0.678070495, 0.416043846, 0.583956154]]), atol=1e-5))
    assert np.all(np.isclose(result2, np.array([[0.416043846, 0.583956154]]), atol=1e-5))

def test_multiple_runs():
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

    results = []

    for i in range(1000):
        results.append(net.run(np.array([[0.5], [-0.5]])))

    for i in range(999):
        assert np.array_equal(results[i], results[i + 1])


def test_faster_run():
    random.seed(1001)
    np.random.seed(1001)
    pop = generate_population(get_default_hrange(), 100, 2, 3, 2)

    input = np.array([[0.1], [0.5]])

    for i in range(3, len(pop)):
        net = pop[i]
        res1 = net.run_normal(input)
        net2 = net.copy()
        res2 = net2.run_faster(input)

        assert np.array_equal(res1, res2)

# test_cn_run()
# test_faster_run()
