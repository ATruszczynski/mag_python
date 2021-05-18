from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *


def test_init_no_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=0, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, batchSize=2, seed=1001)
    assert len(network.neuronCounts) == 2
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 3
    assert len(network.actFuns) == 2
    assert network.actFuns[0] is None
    assert isinstance(network.actFuns[1], Softmax)
    assert isinstance(network.lossFun, QuadDiff)
    assert network.learningRate == 10
    assert network.momCoeffL == 100
    assert network.batchSize == 4

    assert network.layerCount == 2

    assert len(network.weights) == 2
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weights[1], np.array([[0.071019, -0.23531], [0.090597, -1.40620], [1.281623, 0.827808]]), atol=1e-5))

    assert len(network.biases) == 2
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.biases[1], np.array([[0], [0], [0]]), atol=1e-5))

    assert len(network.weight_mom) == 2
    assert np.array_equal(network.weight_mom[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weight_mom[1], np.array([[0, 0], [0, 0], [0, 0]]), atol=1e-5))

    assert len(network.biases_mom) == 2
    assert np.array_equal(network.biases_mom[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.biases_mom[1], np.array([[0], [0], [0]]), atol=1e-5))



def test_init_2_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=2, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, batchSize=0, seed=1001)

    assert len(network.neuronCounts) == 4
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 4
    assert network.neuronCounts[2] == 4
    assert network.neuronCounts[3] == 3
    assert len(network.actFuns) == 4
    assert network.actFuns[0] is None
    assert isinstance(network.actFuns[1], ReLu)
    assert isinstance(network.actFuns[2], ReLu)
    assert isinstance(network.actFuns[3], Softmax)
    assert isinstance(network.lossFun, QuadDiff)
    assert network.learningRate == 10
    assert network.momCoeffL == 100
    assert network.batchSize == 1

    assert network.layerCount == 4

    assert len(network.weights) == 4
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weights[1], np.array([[0.071019, -0.23531], [0.09059, -1.40620], [1.28162, 0.82780], [0.16157, -0.70421]]), atol=1e-5))
    assert np.all(np.isclose(network.weights[2], np.array([[-0.50992, 0.51331, -0.52928, 0.37987],
                                                           [-0.12572, -0.10110, -0.06362, -0.78248],
                                                           [0.69932, -0.37211, -0.45611, -0.67820],
                                                           [0.22578, 0.54550, 0.90252, -0.69979]]), atol=1e-5))
    assert np.all(np.isclose(network.weights[3], np.array([[1.01778, 0.12202, -0.33471, -0.38592],
                                                           [0.68398, 0.72451, -0.01598, 0.67010],
                                                           [0.56757, 0.29046, -0.33311, 0.26854]]), atol=1e-5))

    assert len(network.biases) == 4
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.biases[1], np.array([[0], [0], [0], [0]]), atol=1e-5))
    assert np.all(np.isclose(network.biases[2], np.array([[0], [0], [0], [0]]), atol=1e-5))
    assert np.all(np.isclose(network.biases[3], np.array([[0], [0], [0]]), atol=1e-5))

    assert len(network.weight_mom) == 4
    assert np.array_equal(network.weight_mom[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weight_mom[1], np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), atol=1e-5))
    assert np.all(np.isclose(network.weight_mom[2], np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(network.weight_mom[3], np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), atol=1e-5))

    assert len(network.biases_mom) == 4
    assert np.array_equal(network.biases_mom[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.biases_mom[1], np.array([[0], [0], [0], [0]]), atol=1e-5))
    assert np.all(np.isclose(network.biases_mom[2], np.array([[0], [0], [0], [0]]), atol=1e-5))
    assert np.all(np.isclose(network.biases_mom[3], np.array([[0], [0], [0]]), atol=1e-5))

# random.seed(1001)
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print(random.gauss(0, 1 / sqrt(2)))
# print("-------------------")
#
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print("-------------------")
#
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))
# print(random.gauss(0, 1 / sqrt(4)))

