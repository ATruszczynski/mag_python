from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *


def test_init_no_hidden():
    wei = [np.array([[1, 2], [3, 4], [5, 6]])]
    bia = [np.array([[7], [8], [9]])]

    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hidden_neuron_counts=[], actFuns=[Softmax()],
                                       weights=wei, biases=bia,
                                       seed=1001)

    wei[0][0, 1] = 20
    wei[0][2, 0] = 50
    bia[0][1] = 80

    assert np.array_equal(wei[0], np.array([[1, 20], [3, 4], [50, 6]]))
    assert np.array_equal(bia[0], np.array([[7], [80], [9]]))

    assert len(network.neuronCounts) == 2
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 3
    assert len(network.actFuns) == 2
    assert network.actFuns[0] is None
    assert network.actFuns[1].to_string() == Softmax().to_string()

    assert network.layerCount == 2

    assert len(network.weights) == 2
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.array_equal(network.weights[1], np.array([[1, 2], [3, 4], [5, 6]]))

    assert len(network.biases) == 2
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.array_equal(network.biases[1], np.array([[7], [8], [9]]))

    assert len(network.inp) == 2
    assert np.array_equal(network.inp[0], np.empty((0, 0)))
    assert np.array_equal(network.inp[1], np.empty((0, 0)))

    assert len(network.act) == 2
    assert np.array_equal(network.act[0], np.empty((0, 0)))
    assert np.array_equal(network.act[1], np.empty((0, 0)))

def test_init_2_hidden():
    wei = [np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
           np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]),
           np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]])]
    bia = [np.array([[30], [31], [32], [33]]),
           np.array([[34], [35], [36]]),
           np.array([[37], [38], [39]])]

    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hidden_neuron_counts=[4, 3],
                                       actFuns=[Softmax(), ReLu(), Sigmoid()],
                                       weights=wei,
                                       biases=bia,
                                       seed=1001)

    wei[1][1, 3] = 160
    wei[2][2, 2] = 290
    bia[0][2] = 320
    bia[1][0] = 340

    assert np.array_equal(wei[1], np.array([[9, 10, 11, 12], [13, 14, 15, 160], [17, 18, 19, 20]]))
    assert np.array_equal(wei[2], np.array([[21, 22, 23], [24, 25, 26], [27, 28, 290]]))
    assert np.array_equal(bia[0], np.array([[30], [31], [320], [33]]))
    assert np.array_equal(bia[1], np.array([[340], [35], [36]]))

    assert len(network.neuronCounts) == 4
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 4
    assert network.neuronCounts[2] == 3
    assert network.neuronCounts[3] == 3
    assert len(network.actFuns) == 4
    assert network.actFuns[0] is None
    assert network.actFuns[1].to_string() == Softmax().to_string()
    assert network.actFuns[2].to_string() == ReLu().to_string()
    assert network.actFuns[3].to_string() == Sigmoid().to_string()

    assert network.layerCount == 4

    assert len(network.weights) == 4
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.array_equal(network.weights[1], np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert np.array_equal(network.weights[2], np.array([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]))
    assert np.array_equal(network.weights[3], np.array([[21, 22, 23], [24, 25, 26], [27, 28, 29]]))

    assert len(network.biases) == 4
    assert np.array_equal(network.biases[0], np.empty((0, 0)))
    assert np.array_equal(network.biases[1], np.array([[30], [31], [32], [33]]))
    assert np.array_equal(network.biases[2], np.array([[34], [35], [36]]))
    assert np.array_equal(network.biases[3], np.array([[37], [38], [39]]))

    assert len(network.inp) == 4
    assert np.array_equal(network.inp[0], np.empty((0, 0)))
    assert np.array_equal(network.inp[1], np.empty((0, 0)))
    assert np.array_equal(network.inp[2], np.empty((0, 0)))
    assert np.array_equal(network.inp[3], np.empty((0, 0)))

    assert len(network.act) == 4
    assert np.array_equal(network.act[0], np.empty((0, 0)))
    assert np.array_equal(network.act[1], np.empty((0, 0)))
    assert np.array_equal(network.act[2], np.empty((0, 0)))
    assert np.array_equal(network.act[3], np.empty((0, 0)))


test_init_no_hidden()

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

