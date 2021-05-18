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
    hrange = HyperparameterRange((0, 4), (2, 10), [ReLu(), Sigmoid(), Softmax()], [Softmax(), Sigmoid(), TanH(), ReLu()],
                                 [QuadDiff(), CrossEntropy()], (0, 5), (5, 10), (0, 8))

    random.seed(2020)
    population = generate_population(hrange=hrange, count=2, input_size=10, output_size=5)

    assert population[0].hiddenLayerCount == 4
    assert population[0].neuronCount == pytest.approx(6.955928)
    assert population[0].actFun.to_string() == Softmax().to_string()
    assert population[0].aggrFun.to_string() == ReLu().to_string()
    assert population[0].lossFun.to_string() == CrossEntropy().to_string()
    assert population[0].learningRate == pytest.approx(2.370804)
    assert population[0].momCoeff == pytest.approx(9.668200)
    assert population[0].batchSize == pytest.approx(4.343884)

    assert population[1].hiddenLayerCount == 4
    assert population[1].neuronCount == pytest.approx(2.733362)
    assert population[1].actFun.to_string() == ReLu().to_string()
    assert population[1].aggrFun.to_string() == Sigmoid().to_string()
    assert population[1].lossFun.to_string() == CrossEntropy().to_string()
    assert population[1].learningRate == pytest.approx(3.351164)
    assert population[1].momCoeff == pytest.approx(8.022619)
    assert population[1].batchSize == pytest.approx(3.771914)

def test_pun_fun():
    args = [-1, -0.01, -0.0000001, 0, 0.0000001, 0.01, 1]

    assert punishment_function(args[0]) == pytest.approx(0, abs=1e-5)
    assert punishment_function(args[1]) == pytest.approx(0.566311003, abs=1e-5)
    assert punishment_function(args[2]) == pytest.approx(0.749998125, abs=1e-5)
    assert punishment_function(args[3]) == pytest.approx(0.75, abs=1e-5)
    assert punishment_function(args[4]) == pytest.approx(1.250001875, abs=1e-5)
    assert punishment_function(args[5]) == pytest.approx(1.433688997, abs=1e-5)
    assert punishment_function(args[6]) == pytest.approx(2, abs=1e-5)

def test_get_in_radius():
    random.seed(2020)

    p = get_in_radius(1, 0, 5, 0.5)
    assert p == pytest.approx(2.168842)

    p = get_in_radius(1, 0, 5, 0.75)
    assert p == pytest.approx(0.828988)


random.seed(2020)
print(random.uniform(0, 3.5))
print(random.uniform(0, 4.75))

