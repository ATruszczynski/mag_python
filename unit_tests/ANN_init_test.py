from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *


def test_init_no_hidden():
    network = FeedForwardNeuralNetwork(neuronCounts=[2, 3], actFun=[Sigmoid()], lossFun=QuadDiff(), learningRate=-3,
                                       momCoeffL=-2, batchSize=-4, seed=1001)

    assert len(network.neuronCounts) == 2
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 3

    assert len(network.actFuns) == 2
    assert network.actFuns[0] is None
    assert network.actFuns[1].to_string() == Sigmoid().to_string()

    assert network.layerCount == 2

    assert len(network.weights) == 2
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weights[1], np.array([[-0.7682336, -0.63361373], [-0.21658636, -0.94747618], [-0.85318485, -0.45376938]]), atol=1e-3))

    assert len(network.biases) == 2
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.array_equal(network.biases[1], np.array([[0], [0], [0]]))

    assert len(network.weight_mom) == 2
    assert np.array_equal(network.weight_mom[0], np.empty((0, 0)))
    assert np.array_equal(network.weight_mom[1], np.zeros((3, 2)))

    assert len(network.biases_mom) == 2
    assert np.array_equal(network.biases_mom[0], np.empty((0, 0)))
    assert np.array_equal(network.biases_mom[1], np.zeros((3, 1)))

    assert len(network.inp) == 2
    assert np.array_equal(network.inp[0], np.empty((0, 0)))
    assert np.array_equal(network.inp[1], np.empty((0, 0)))

    assert len(network.act) == 2
    assert np.array_equal(network.act[0], np.empty((0, 0)))
    assert np.array_equal(network.act[1], np.empty((0, 0)))

    assert network.lossFun.to_string() == QuadDiff().to_string()
    assert network.learningRate == 0.001
    assert network.momCoeffL == 0.01
    assert network.batchSize == 1 / 16

def test_init_2_hidden():
    network = FeedForwardNeuralNetwork(neuronCounts=[2, 4, 3], actFun=[Sigmoid(), ReLu()], lossFun=CrossEntropy(), learningRate=-1,
                                       momCoeffL=-3, batchSize=-2, seed=1001)

    assert len(network.neuronCounts) == 3
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 4
    assert network.neuronCounts[2] == 3

    assert len(network.actFuns) == 3
    assert network.actFuns[0] is None
    assert network.actFuns[1].to_string() == Sigmoid().to_string()
    assert network.actFuns[2].to_string() == ReLu().to_string()

    assert network.layerCount == 3

    assert len(network.weights) == 3
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weights[1], np.array([[-0.7682336, -0.63361373], [-0.21658636, -0.94747618], [-0.85318485, -0.45376938], [ 0.92485723, 1.30493758]]), atol=1e-4))
    assert np.all(np.isclose(network.weights[2], np.array([[0.41455747, -0.01164941, -0.10428198, -0.45830987], [-0.53737129, -0.04307174,  0.58791927, -0.81754587],
                                                       [0.614097, 0.53819313, 0.19738643, -0.19385041]]), atol=1e-4))

    assert len(network.biases) == 3
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.array_equal(network.biases[1], np.array([[0], [0], [0], [0]]))
    assert np.array_equal(network.biases[2], np.array([[0], [0], [0]]))

    assert len(network.weight_mom) == 3
    assert np.array_equal(network.weight_mom[0], np.empty((0, 0)))
    assert np.array_equal(network.weight_mom[1], np.zeros((4, 2)))
    assert np.array_equal(network.weight_mom[2], np.zeros((3, 4)))

    assert len(network.biases_mom) == 3
    assert np.array_equal(network.biases_mom[0], np.empty((0, 0)))
    assert np.array_equal(network.biases_mom[1], np.zeros((4, 1)))
    assert np.array_equal(network.biases_mom[2], np.zeros((3, 1)))

    assert len(network.inp) == 3
    assert np.array_equal(network.inp[0], np.empty((0, 0)))
    assert np.array_equal(network.inp[1], np.empty((0, 0)))
    assert np.array_equal(network.inp[2], np.empty((0, 0)))

    assert len(network.act) == 3
    assert np.array_equal(network.act[0], np.empty((0, 0)))
    assert np.array_equal(network.act[1], np.empty((0, 0)))
    assert np.array_equal(network.act[2], np.empty((0, 0)))

    assert network.lossFun.to_string() == CrossEntropy().to_string()
    assert network.learningRate == 0.1
    assert network.momCoeffL == 0.001
    assert network.batchSize == 1 / 4


# random.seed(1001)
# np.random.seed(1001)
# print(np.random.normal(0, 1 / sqrt(2), (3, 2)))
# print(np.random.normal(0, 1 / sqrt(3), (3, 3)))

# random.seed(1001)
# np.random.seed(1001)
# print(np.random.normal(0, 1 / sqrt(2), (4, 2)))
# print(np.random.normal(0, 1 / sqrt(4), (3, 4)))

