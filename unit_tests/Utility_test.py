from neural_network.FeedForwardNeuralNetwork import network_from_point
from utility.Utility import *
import pytest

def test_batch_divide_round():
    inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14]), np.array([15])]

    batches = divideIntoBatches(inputs, outputs, 2)

    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2

    assert np.array_equal(batches[0][0][0], np.array([0]))
    assert np.array_equal(batches[0][0][1], np.array([10]))
    assert np.array_equal(batches[0][1][0], np.array([1]))
    assert np.array_equal(batches[0][1][1], np.array([11]))

    assert np.array_equal(batches[1][0][0], np.array([2]))
    assert np.array_equal(batches[1][0][1], np.array([12]))
    assert np.array_equal(batches[1][1][0], np.array([3]))
    assert np.array_equal(batches[1][1][1], np.array([13]))

    assert np.array_equal(batches[2][0][0], np.array([4]))
    assert np.array_equal(batches[2][0][1], np.array([14]))
    assert np.array_equal(batches[2][1][0], np.array([5]))
    assert np.array_equal(batches[2][1][1], np.array([15]))



def test_batch_divide_not_round():
    inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14])]

    batches = divideIntoBatches(inputs, outputs, 2)

    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1

    assert np.array_equal(batches[0][0][0], np.array([0]))
    assert np.array_equal(batches[0][0][1], np.array([10]))
    assert np.array_equal(batches[0][1][0], np.array([1]))
    assert np.array_equal(batches[0][1][1], np.array([11]))

    assert np.array_equal(batches[1][0][0], np.array([2]))
    assert np.array_equal(batches[1][0][1], np.array([12]))
    assert np.array_equal(batches[1][1][0], np.array([3]))
    assert np.array_equal(batches[1][1][1], np.array([13]))

    assert np.array_equal(batches[2][0][0], np.array([4]))
    assert np.array_equal(batches[2][0][1], np.array([14]))

def test_try_choose_different_possible():
    options = [ReLu(), Sigmoid(), Softmax()]
    curr = Sigmoid()

    random.seed(2020) # 0, 1

    res = try_choose_different(current=curr, possibilities=options)
    assert res.to_string() == ReLu().to_string()

    res = try_choose_different(current=curr, possibilities=options)
    assert res.to_string() == Softmax().to_string()

    options = [1, 2, 5]
    curr = 5

    random.seed(2020) # 0, 1

    res = try_choose_different(current=curr, possibilities=options)
    assert res == 1

    res = try_choose_different(current=curr, possibilities=options)
    assert res == 2

def test_try_choose_different_impossible():
    options = [Sigmoid()]
    curr = Sigmoid()

    random.seed(2020) # 0, 1

    res = try_choose_different(current=curr, possibilities=options)
    assert res.to_string() == Sigmoid().to_string()

    options = [1]
    curr = 1

    random.seed(2020) # 0, 1

    res = try_choose_different(current=curr, possibilities=options)
    assert res == 1

def test_choose_without_repetition():
    options = [1, 2, 3, 4]

    random.seed(2020) # 0, 1
    chosen = choose_without_repetition(options=options, count=2)

    assert chosen[0] == 2
    assert chosen[1] == 4

def test_ohe():
    labels = [1, 2, 0, 4, 2, 1, 3]
    ohe = one_hot_endode(labels)

    assert np.array_equal(ohe[0], np.array([[0],[1],[0],[0],[0]]))
    assert np.array_equal(ohe[1], np.array([[0],[0],[1],[0],[0]]))
    assert np.array_equal(ohe[2], np.array([[1],[0],[0],[0],[0]]))
    assert np.array_equal(ohe[3], np.array([[0],[0],[0],[0],[1]]))
    assert np.array_equal(ohe[4], np.array([[0],[0],[1],[0],[0]]))
    assert np.array_equal(ohe[5], np.array([[0],[1],[0],[0],[0]]))
    assert np.array_equal(ohe[6], np.array([[0],[0],[0],[1],[0]]))

def test_generate_population():
    hrange = HyperparameterRange((0, 4), (0, 10), [ReLu(), Sigmoid(), Softmax()], 1, 2)

    random.seed(1111)
    np.random.seed(1111)
    population = generate_population(hrange=hrange, count=2, input_size=2, output_size=3)

    assert population[0].input_size == 2
    assert population[0].output_size == 3
    assert len(population[0].hidden_neuron_counts) == 1
    assert population[0].hidden_neuron_counts[0] == 3
    assert population[0].activation_functions[0].to_string() == Sigmoid().to_string()
    assert population[0].activation_functions[1].to_string() == Softmax().to_string()
    assert np.all(np.isclose(population[0].weights[0], np.array([[-0.37904612, -0.99598032], [-0.52881055, -0.52441656], [0.47183175, -0.00906385]]), atol=1e-5))
    assert np.all(np.isclose(population[0].weights[1], np.array([[-0.06775805, -0.52573575, -0.12968164], [-0.51265697, -0.23232018, 0.67678738], [0.31036945, -0.70310665, 0.27829033]]), atol=1e-5))
    assert np.all(np.isclose(population[0].biases[0], np.array([[-1.6178032], [1.70001481], [-0.62570631]]), atol=1e-5))
    assert np.all(np.isclose(population[0].biases[1], np.array([[1.1377014], [-1.49397476], [0.42659728]]), atol=1e-5))

# def test_pun_fun():
#     args = [-1, -0.01, -0.0000001, 0, 0.0000001, 0.01, 1]
#
#     assert punishment_function(args[0]) == pytest.approx(0, abs=1e-5)
#     assert punishment_function(args[1]) == pytest.approx(0.566311003, abs=1e-5)
#     assert punishment_function(args[2]) == pytest.approx(0.749998125, abs=1e-5)
#     assert punishment_function(args[3]) == pytest.approx(0.75, abs=1e-5)
#     assert punishment_function(args[4]) == pytest.approx(1.250001875, abs=1e-5)
#     assert punishment_function(args[5]) == pytest.approx(1.433688997, abs=1e-5)
#     assert punishment_function(args[6]) == pytest.approx(2, abs=1e-5)

def test_get_in_radius():
    random.seed(2020)

    p = get_in_radius(1, 0, 5, 0.5)
    assert p == pytest.approx(2.168842)

    p = get_in_radius(1, 0, 5, 0.75)
    assert p == pytest.approx(0.828988)

# def test_get_network_from_point():
#     point = AnnPoint(inputSize=2, outputSize=3, hiddenLayerCount=2, neuronCount=2, actFun=ReLu(),
#                      aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeff=2, batchSize=0)
#     network = network_from_point(point, 1001)
#     TODO fix
#     assert len(network.neuronCounts) == 4
#     assert network.neuronCounts[0] == 2
#     assert network.neuronCounts[1] == 4
#     assert network.neuronCounts[2] == 4
#     assert network.neuronCounts[3] == 3
#     assert len(network.actFuns) == 4
#     assert network.actFuns[0] is None
#     assert isinstance(network.actFuns[1], ReLu)
#     assert isinstance(network.actFuns[2], ReLu)
#     assert isinstance(network.actFuns[3], Softmax)
#     assert isinstance(network.lossFun, QuadDiff)
#     assert network.learningRate == 10
#     assert network.momCoeffL == 100
#     assert network.batchSize == 1
#
#     assert network.layerCount == 4
#
#     assert len(network.weights) == 4
#     assert np.array_equal(network.weights[0], np.empty((0, 0)))
#     assert np.all(np.isclose(network.weights[1], np.array([[0.071019, -0.23531], [0.09059, -1.40620], [1.28162, 0.82780], [0.16157, -0.70421]]), atol=1e-5))
#     assert np.all(np.isclose(network.weights[2], np.array([[-0.50992, 0.51331, -0.52928, 0.37987],
#                                                            [-0.12572, -0.10110, -0.06362, -0.78248],
#                                                            [0.69932, -0.37211, -0.45611, -0.67820],
#                                                            [0.22578, 0.54550, 0.90252, -0.69979]]), atol=1e-5))
#     assert np.all(np.isclose(network.weights[3], np.array([[1.01778, 0.12202, -0.33471, -0.38592],
#                                                            [0.68398, 0.72451, -0.01598, 0.67010],
#                                                            [0.56757, 0.29046, -0.33311, 0.26854]]), atol=1e-5))
#
#     assert len(network.biases) == 4
#     assert np.array_equal(network.biases[0], np.empty((0, 0)))
#     assert np.all(np.isclose(network.biases[1], np.array([[0], [0], [0], [0]]), atol=1e-5))
#     assert np.all(np.isclose(network.biases[2], np.array([[0], [0], [0], [0]]), atol=1e-5))
#     assert np.all(np.isclose(network.biases[3], np.array([[0], [0], [0]]), atol=1e-5))
#
#     assert len(network.weight_mom) == 4
#     assert np.array_equal(network.weight_mom[0], np.empty((0, 0)))
#     assert np.all(np.isclose(network.weight_mom[1], np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), atol=1e-5))
#     assert np.all(np.isclose(network.weight_mom[2], np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), atol=1e-5))
#     assert np.all(np.isclose(network.weight_mom[3], np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), atol=1e-5))
#
#     assert len(network.biases_mom) == 4
#     assert np.array_equal(network.biases_mom[0], np.empty((0, 0)))
#     assert np.all(np.isclose(network.biases_mom[1], np.array([[0], [0], [0], [0]]), atol=1e-5))
#     assert np.all(np.isclose(network.biases_mom[2], np.array([[0], [0], [0], [0]]), atol=1e-5))
#     assert np.all(np.isclose(network.biases_mom[3], np.array([[0], [0], [0]]), atol=1e-5))


random.seed(1111)
np.random.seed(1111)
print(random.randint(0, 4))
print("")

print(random.randint(0, 10))
print("")

print(random.randint(0, 2))
print(random.randint(0, 2))
print("")

# 2 3 2
print(np.random.uniform(-2, 2, size=(3, 1)))
print(np.random.uniform(-1, 1, size=(3, 2)))
print("")

print(np.random.uniform(-2, 2, size=(3, 1)))
print(np.random.uniform(-1, 1, size=(3, 3)))

print(random.randint(0, 4))

test_generate_population()

