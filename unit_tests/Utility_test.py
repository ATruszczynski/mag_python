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

    assert np.array_equal(ohe[0], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(ohe[1], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(ohe[2], np.array([[1], [0], [0], [0], [0]]))
    assert np.array_equal(ohe[3], np.array([[0], [0], [0], [0], [1]]))
    assert np.array_equal(ohe[4], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(ohe[5], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(ohe[6], np.array([[0], [0], [0], [1], [0]]))

def test_generate_population():
    hrange = HyperparameterRange((0, 2), (0, 10), [ReLu(), Sigmoid(), Softmax()], [CrossEntropy(), QuadDiff()],
                                 (-1, 0), (-3, -2), (-5, -4))

    random.seed(1010)
    np.random.seed(1010)
    population = generate_population(hrange=hrange, count=2, input_size=2, output_size=3)

    assert len(population[0].neuronCounts) == 4
    assert population[0].neuronCounts[0] == 2
    assert population[0].neuronCounts[1] == 9
    assert population[0].neuronCounts[2] == 8
    assert population[0].neuronCounts[3] == 3

    assert len(population[0].actFuns) == 3
    assert population[0].actFuns[0].to_string() == ReLu().to_string()
    assert population[0].actFuns[1].to_string() == ReLu().to_string()
    assert population[0].actFuns[2].to_string() == Sigmoid().to_string()

    assert population[0].lossFun.to_string() == QuadDiff().to_string()

    assert population[0].learningRate == pytest.approx(-0.83566, abs=1e-4)
    assert population[0].momCoeff == pytest.approx(-2.56324, abs=1e-4)
    assert population[0].batchSize == pytest.approx(-4.27116, abs=1e-4)



    assert len(population[1].neuronCounts) == 2
    assert population[1].neuronCounts[0] == 2
    assert population[1].neuronCounts[1] == 3

    assert len(population[1].actFuns) == 1
    assert population[1].actFuns[0].to_string() == Softmax().to_string()

    assert population[1].lossFun.to_string() == QuadDiff().to_string()

    assert population[1].learningRate == pytest.approx(-0.00331, abs=1e-4)
    assert population[1].momCoeff == pytest.approx(-2.42444, abs=1e-4)
    assert population[1].batchSize == pytest.approx(-4.85385, abs=1e-4)

def test_pun_fun():
    args = [-1, -0.01, -0.0000001, 0, 0.0000001, 0.01, 1]

    assert punishment_function(args[0]) == pytest.approx(0, abs=1e-5)
    assert punishment_function(args[1]) == pytest.approx(0.283155502, abs=1e-5)
    assert punishment_function(args[2]) == pytest.approx(0.374999063, abs=1e-5)
    assert punishment_function(args[3]) == pytest.approx(0.375, abs=1e-5)
    assert punishment_function(args[4]) == pytest.approx(0.625000938, abs=1e-5)
    assert punishment_function(args[5]) == pytest.approx(0.716844498, abs=1e-5)
    assert punishment_function(args[6]) == pytest.approx(1, abs=1e-5)

def test_get_in_radius():
    random.seed(2020)

    p = get_in_radius(1, 0, 5, 0.5)
    assert p == pytest.approx(2.168842)

    p = get_in_radius(1, 0, 5, 0.75)
    assert p == pytest.approx(0.828988)

def test_get_network_from_point():
    point = AnnPoint(neuronCounts=[2, 4, 4, 3], actFuns=[ReLu(), ReLu(), Softmax()], lossFun=QuadDiff(), learningRate=1, momCoeff=2, batchSize=3)
    network = network_from_point(point, 1010)

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
    assert network.batchSize == 8

    assert network.layerCount == 4

    assert len(network.weights) == 4
    assert np.array_equal(network.weights[0], np.empty((0, 0)))
    assert np.all(np.isclose(network.weights[1], np.array([[-0.83116718, -0.27092632], [-1.040413, -1.27319421],[0.09199489, 1.12827275], [0.70227065, -1.67139339]]), atol=1e-5))
    assert np.all(np.isclose(network.weights[2], np.array([[-0.23979613, -0.82519097, -0.27174483,  0.38980573],
                                                           [-0.25130439,  0.14294476,  1.35161869, -0.03725841],
                                                          [-0.68505133,  0.17929361,  0.29902494, -0.1533996],
                                                            [0.38148435, -0.4994389,   0.50463211, -0.12389273]]), atol=1e-5))
    assert np.all(np.isclose(network.weights[3], np.array([[-1.2917587,   0.37861067, -0.88823027, -0.38984775],
                                                           [-0.45926747, -0.19749436,  0.12400293,  0.14901888],
                                                          [ 0.14190831, -0.23561145,  0.47601405, -0.31930159]]), atol=1e-5))

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


# random.seed(1010)
# np.random.seed(1010)
# print(random.randint(0, 2))
# print("")
#
# print(random.randint(0, 10))
# print(random.randint(0, 2))
#
# print(random.randint(0, 10))
# print(random.randint(0, 2))
#
# print(random.randint(0, 10))
# print(random.randint(0, 2))
# print("")
#
# print(random.randint(0, 1))
# print(random.uniform(-1, 0))
# print(random.uniform(-3, -2))
# print(random.uniform(-5, -4))
# print("")
#
# print("next")
#
# print(random.randint(0, 2))
# print("")
#
# print(random.randint(0, 10))
# print(random.randint(0, 2))
# print("")
#
# print(random.randint(0, 1))
# print(random.uniform(-1, 0))
# print(random.uniform(-3, -2))
# print(random.uniform(-5, -4))
# print("")



# # 2 3 2
# print(np.random.uniform(-2, 2, size=(3, 1)))
# print(np.random.uniform(-1, 1, size=(3, 2)))
# print("")
#
# print(np.random.uniform(-2, 2, size=(3, 1)))
# print(np.random.uniform(-1, 1, size=(3, 3)))
#
# print(random.randint(0, 4))

# test_generate_population()

# random.seed(1010)
# np.random.seed(1010)
# print(np.random.normal(0, 1 / sqrt(2), (4, 2)))
# print(np.random.normal(0, 1 / sqrt(4), (4, 4)))
# print(np.random.normal(0, 1 / sqrt(4), (3, 4)))
#
# test_get_network_from_point()
