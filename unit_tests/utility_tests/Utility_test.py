from utility.Utility import *
import pytest

# def test_batch_divide_round():
#     inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
#     outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14]), np.array([15])]
#
#     batches = divideIntoBatches(inputs, outputs, 2)
#
#     assert len(batches) == 3
#
#     assert len(batches[0]) == 2
#     assert len(batches[1]) == 2
#     assert len(batches[2]) == 2
#
#     assert np.array_equal(batches[0][0][0], np.array([0]))
#     assert np.array_equal(batches[0][0][1], np.array([10]))
#     assert np.array_equal(batches[0][1][0], np.array([1]))
#     assert np.array_equal(batches[0][1][1], np.array([11]))
#
#     assert np.array_equal(batches[1][0][0], np.array([2]))
#     assert np.array_equal(batches[1][0][1], np.array([12]))
#     assert np.array_equal(batches[1][1][0], np.array([3]))
#     assert np.array_equal(batches[1][1][1], np.array([13]))
#
#     assert np.array_equal(batches[2][0][0], np.array([4]))
#     assert np.array_equal(batches[2][0][1], np.array([14]))
#     assert np.array_equal(batches[2][1][0], np.array([5]))
#     assert np.array_equal(batches[2][1][1], np.array([15]))


#
# def test_batch_divide_not_round():
#     inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
#     outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14])]
#
#     batches = divideIntoBatches(inputs, outputs, 2)
#
#     assert len(batches) == 3
#
#     assert len(batches[0]) == 2
#     assert len(batches[1]) == 2
#     assert len(batches[2]) == 1
#
#     assert np.array_equal(batches[0][0][0], np.array([0]))
#     assert np.array_equal(batches[0][0][1], np.array([10]))
#     assert np.array_equal(batches[0][1][0], np.array([1]))
#     assert np.array_equal(batches[0][1][1], np.array([11]))
#
#     assert np.array_equal(batches[1][0][0], np.array([2]))
#     assert np.array_equal(batches[1][0][1], np.array([12]))
#     assert np.array_equal(batches[1][1][0], np.array([3]))
#     assert np.array_equal(batches[1][1][1], np.array([13]))
#
#     assert np.array_equal(batches[2][0][0], np.array([4]))
#     assert np.array_equal(batches[2][0][1], np.array([14]))

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

# def test_generate_population():
#     hrange = HyperparameterRange((0, 2), (0, 10), [ReLu(), Sigmoid(), Softmax()], [CrossEntropy(), QuadDiff()],
#                                  (-1, 0), (-3, -2), (-5, -4))
#
#     random.seed(1010)
#     np.random.seed(1010)
#     population = generate_population(hrange=hrange, count=2, input_size=2, output_size=3)
#
#     assert len(population[0].neuronCounts) == 4
#     assert population[0].neuronCounts[0] == 2
#     assert population[0].neuronCounts[1] == 9
#     assert population[0].neuronCounts[2] == 8
#     assert population[0].neuronCounts[3] == 3
#
#     assert len(population[0].actFuns) == 3
#     assert population[0].actFuns[0].to_string() == ReLu().to_string()
#     assert population[0].actFuns[1].to_string() == ReLu().to_string()
#     assert population[0].actFuns[2].to_string() == Sigmoid().to_string()
#
#     assert population[0].lossFun.to_string() == QuadDiff().to_string()
#
#     assert population[0].learningRate == pytest.approx(-0.83566, abs=1e-4)
#     assert population[0].momCoeff == pytest.approx(-2.56324, abs=1e-4)
#     assert population[0].batchSize == pytest.approx(-4.27116, abs=1e-4)
#
#
#
#     assert len(population[1].neuronCounts) == 2
#     assert population[1].neuronCounts[0] == 2
#     assert population[1].neuronCounts[1] == 3
#
#     assert len(population[1].actFuns) == 1
#     assert population[1].actFuns[0].to_string() == Softmax().to_string()
#
#     assert population[1].lossFun.to_string() == QuadDiff().to_string()
#
#     assert population[1].learningRate == pytest.approx(-0.00331, abs=1e-4)
#     assert population[1].momCoeff == pytest.approx(-2.42444, abs=1e-4)
#     assert population[1].batchSize == pytest.approx(-4.85385, abs=1e-4)

def test_generate_population_limits():
    hrange = HyperparameterRange((0, 2), (0, 5), (1, 5), (10, 20), [ReLu(), Sigmoid(), Softmax()], mut_radius=(-1, 0),
                                 depr=(-2, -1), multi=(-3, -2), p_prob=(-4, -3), c_prob=(-5, -4),
                                 p_rad=(-6, -5))

    random.seed(1001)
    n = 200
    pop = generate_population(hrange, n, 10, 20)

    input_sizes = [pop[i].input_size for i in range(len(pop))]
    output_sizes = [pop[i].output_size for i in range(len(pop))]
    hidden_sizes = [pop[i].hidden_end_index - pop[i].hidden_start_index for i in range(len(pop))]
    its = [pop[i].net_it for i in range(len(pop))]

    assert max(input_sizes) == min(input_sizes)
    assert max(input_sizes) == 10
    assert max(output_sizes) == min(output_sizes)
    assert max(output_sizes) == 20
    assert max(hidden_sizes) == 20
    assert min(hidden_sizes) == 10
    assert min(its) == 1
    assert max(its) == 5

    all_weights = []
    all_biases = []
    all_act_funs = []
    all_aggr_funs = []

    all_mut_rad = []
    all_p_mut = []
    all_c_prob = []

    all_wb_mut = []
    all_s_mut = []
    all_r_prob = []

    for i in range(len(pop)):
        net = pop[i]
        links = net.links

        assert links.shape[0] == net.neuron_count
        assert links.shape[1] == net.neuron_count
        assert net.weights.shape[0] == net.neuron_count
        assert net.weights.shape[1] == net.neuron_count
        assert net.biases.shape[0] == 1
        assert net.biases.shape[1] == net.neuron_count

        non_zero_ind = np.where(links != 0)

        for j in range(len(net.actFuns)):
            if net.actFuns[j] is not None:
                all_act_funs.append(net.actFuns[j].copy())
        all_aggr_funs.append(net.aggrFun.copy())

        weights = net.weights.copy()
        weights[non_zero_ind] = 0
        assert np.max(weights) == 0
        assert np.min(weights) == 0

        for j in range(net.biases.shape[1]):
            if not np.isnan(net.biases[0, j]):
                all_biases.append(net.biases[0, j])

        for ind in range(len(non_zero_ind[0])):
            r = non_zero_ind[0][ind]
            c = non_zero_ind[1][ind]
            assert c != r
            assert c >= net.hidden_start_index
            assert r < net.hidden_end_index

            all_weights.append(net.weights[r, c])

        all_mut_rad.append(net.mutation_radius)
        all_p_mut.append(net.p_prob)
        all_c_prob.append(net.c_prob)

        all_wb_mut.append(net.depr)
        all_s_mut.append(net.multi)
        all_r_prob.append(net.p_rad)

    assert max(all_weights) <= 2
    assert max(all_weights) > 1.95
    assert min(all_weights) >= 0
    assert min(all_weights) < 0.05

    assert np.max(all_biases) <= 5
    assert np.max(all_biases) > 4.95
    assert np.min(all_biases) >= 0
    assert np.min(all_biases) < 0.05

    for i in range(len(hrange.actFunSet)):
        afh = hrange.actFunSet[i]
        isthere = False
        for j in range(len(all_act_funs)):
            afp = all_act_funs[j]
            isthere = isthere or afh.to_string() == afp.to_string()
            if isthere == True:
                break
        assert isthere

    for i in range(len(hrange.actFunSet)):
        afh = hrange.actFunSet[i]
        isthere = False
        for j in range(len(all_aggr_funs)):
            afp = all_aggr_funs[j]
            isthere = isthere or afh.to_string() == afp.to_string()
            if isthere == True:
                break
        assert isthere

    # hrange = HyperparameterRange((0, 2), (0, 5), (1, 5), (10, 20), [ReLu(), Sigmoid(), Softmax()], mut_radius=(0, 1),
    #                              sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
    #                              dstr_mut_prob=(0.44, 0.55))

    assert min(all_mut_rad) >= -1
    assert min(all_mut_rad) <= -0.95
    assert max(all_mut_rad) <= 0
    assert max(all_mut_rad) >= -0.05

    assert min(all_wb_mut) >= -2
    assert min(all_wb_mut) <= -1.95
    assert max(all_wb_mut) <= -1
    assert max(all_wb_mut) >= -1.05

    assert min(all_s_mut) >= -3
    assert min(all_s_mut) <= -2.95
    assert max(all_s_mut) <= -2
    assert max(all_s_mut) >= -2.05

    assert min(all_p_mut) >= -4
    assert min(all_p_mut) <= -3.95
    assert max(all_p_mut) <= -3
    assert max(all_p_mut) >= -3.05

    assert min(all_c_prob) >= -5
    assert min(all_c_prob) <= -4.95
    assert max(all_c_prob) <= -4
    assert max(all_c_prob) >= -4.05

    assert min(all_r_prob) >= -6
    assert min(all_r_prob) <= -5.95
    assert max(all_r_prob) <= -5
    assert max(all_r_prob) >= -5.05


def test_generate_population_limits_aggr():
    hrange = HyperparameterRange((0, 2), (0, 5), (1, 5), (10, 20), [ReLu(), Sigmoid(), Softmax()], mut_radius=(-1, 0),
                                 depr=(-2, -1), multi=(-3, -2), p_prob=(-4, -3), c_prob=(-5, -4),
                                 p_rad=(-6, -5), aggrFuns=[LReLu(), SincAct(), Sigmoid()])

    random.seed(1001)
    n = 200
    pop = generate_population(hrange, n, 10, 20)

    input_sizes = [pop[i].input_size for i in range(len(pop))]
    output_sizes = [pop[i].output_size for i in range(len(pop))]
    hidden_sizes = [pop[i].hidden_end_index - pop[i].hidden_start_index for i in range(len(pop))]
    its = [pop[i].net_it for i in range(len(pop))]

    assert max(input_sizes) == min(input_sizes)
    assert max(input_sizes) == 10
    assert max(output_sizes) == min(output_sizes)
    assert max(output_sizes) == 20
    assert max(hidden_sizes) == 20
    assert min(hidden_sizes) == 10
    assert min(its) == 1
    assert max(its) == 5

    all_weights = []
    all_biases = []
    all_act_funs = []
    all_aggr_funs = []

    all_mut_rad = []
    all_p_mut = []
    all_c_prob = []

    all_wb_mut = []
    all_s_mut = []
    all_r_prob = []

    for i in range(len(pop)):
        net = pop[i]
        links = net.links

        assert links.shape[0] == net.neuron_count
        assert links.shape[1] == net.neuron_count
        assert net.weights.shape[0] == net.neuron_count
        assert net.weights.shape[1] == net.neuron_count
        assert net.biases.shape[0] == 1
        assert net.biases.shape[1] == net.neuron_count

        non_zero_ind = np.where(links != 0)

        for j in range(len(net.actFuns)):
            if net.actFuns[j] is not None:
                all_act_funs.append(net.actFuns[j].copy())
        all_aggr_funs.append(net.aggrFun.copy())

        weights = net.weights.copy()
        weights[non_zero_ind] = 0
        assert np.max(weights) == 0
        assert np.min(weights) == 0

        for j in range(net.biases.shape[1]):
            if not np.isnan(net.biases[0, j]):
                all_biases.append(net.biases[0, j])

        for ind in range(len(non_zero_ind[0])):
            r = non_zero_ind[0][ind]
            c = non_zero_ind[1][ind]
            assert c != r
            assert c >= net.hidden_start_index
            assert r < net.hidden_end_index

            all_weights.append(net.weights[r, c])

        all_mut_rad.append(net.mutation_radius)
        all_p_mut.append(net.p_prob)
        all_c_prob.append(net.c_prob)

        all_wb_mut.append(net.depr)
        all_s_mut.append(net.multi)
        all_r_prob.append(net.p_rad)

    assert max(all_weights) <= 2
    assert max(all_weights) > 1.95
    assert min(all_weights) >= 0
    assert min(all_weights) < 0.05

    assert np.max(all_biases) <= 5
    assert np.max(all_biases) > 4.95
    assert np.min(all_biases) >= 0
    assert np.min(all_biases) < 0.05

    for i in range(len(hrange.actFunSet)):
        afh = hrange.actFunSet[i]
        isthere = False
        for j in range(len(all_act_funs)):
            afp = all_act_funs[j]
            isthere = isthere or afh.to_string() == afp.to_string()
            if isthere == True:
                break
        assert isthere

    for i in range(len(hrange.aggrFuns)):
        afh = hrange.aggrFuns[i]
        isthere = False
        for j in range(len(all_aggr_funs)):
            afp = all_aggr_funs[j]
            isthere = isthere or afh.to_string() == afp.to_string()
            if isthere == True:
                break
        assert isthere

    # hrange = HyperparameterRange((0, 2), (0, 5), (1, 5), (10, 20), [ReLu(), Sigmoid(), Softmax()], mut_radius=(0, 1),
    #                              sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
    #                              dstr_mut_prob=(0.44, 0.55))

    assert min(all_mut_rad) >= -1
    assert min(all_mut_rad) <= -0.95
    assert max(all_mut_rad) <= 0
    assert max(all_mut_rad) >= -0.05

    assert min(all_wb_mut) >= -2
    assert min(all_wb_mut) <= -1.95
    assert max(all_wb_mut) <= -1
    assert max(all_wb_mut) >= -1.05

    assert min(all_s_mut) >= -3
    assert min(all_s_mut) <= -2.95
    assert max(all_s_mut) <= -2
    assert max(all_s_mut) >= -2.05

    assert min(all_p_mut) >= -4
    assert min(all_p_mut) <= -3.95
    assert max(all_p_mut) <= -3
    assert max(all_p_mut) >= -3.05

    assert min(all_c_prob) >= -5
    assert min(all_c_prob) <= -4.95
    assert max(all_c_prob) <= -4
    assert max(all_c_prob) >= -4.05

    assert min(all_r_prob) >= -6
    assert min(all_r_prob) <= -5.95
    assert max(all_r_prob) <= -5
    assert max(all_r_prob) >= -5.05


# def test_pun_fun():
#     args = [-1, -0.01, -0.0000001, 0, 0.0000001, 0.01, 1]
#
#     assert punishment_function(args[0]) == pytest.approx(0, abs=1e-5)
#     assert punishment_function(args[1]) == pytest.approx(0.283155502, abs=1e-5)
#     assert punishment_function(args[2]) == pytest.approx(0.374999063, abs=1e-5)
#     assert punishment_function(args[3]) == pytest.approx(0.375, abs=1e-5)
#     assert punishment_function(args[4]) == pytest.approx(0.625000938, abs=1e-5)
#     assert punishment_function(args[5]) == pytest.approx(0.716844498, abs=1e-5)
#     assert punishment_function(args[6]) == pytest.approx(1, abs=1e-5)

def test_get_in_radius():
    random.seed(2020)

    p = get_in_radius(1, 0, 5, 0.5)
    assert p == pytest.approx(2.168842)

    p = get_in_radius(1, 0, 5, 0.75)
    assert p == pytest.approx(0.828988)


def test_get_in_radius_limits():
    random.seed(2020)

    minB = 0
    maxB = 5
    radius = 5

    hist = []
    for i in range(500):
        hist.append(get_in_radius(1, minB, maxB, radius))

    assert min(hist) >= minB
    assert max(hist) <= maxB

def test_copy_list_of_arrays():
    arrays = [np.zeros((2, 2)), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.ones((2, 4))]

    carray = copy_list_of_arrays(arrays=arrays)

    assert np.array_equal(carray[0], np.zeros((2, 2)))
    assert np.array_equal(carray[1], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.array_equal(carray[2], np.ones((2, 4)))

    arrays[0][-1, -1] = 5
    assert np.array_equal(arrays[0], np.array([[0, 0], [0, 5]]))
    assert np.array_equal(carray[0], np.array([[0, 0], [0, 0]]))

    arrays[1][-2, -2] = -1
    assert np.array_equal(arrays[1], np.array([[1, 2, 3], [4, -1, 6], [7, 8, 9]]))
    assert np.array_equal(carray[1], np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    arrays[2][-2, -3] = 0
    assert np.array_equal(arrays[2], np.array([[1, 0, 1, 1], [1, 1, 1, 1]]))
    assert np.array_equal(carray[2], np.array([[1, 1, 1, 1], [1, 1, 1, 1]]))

# def test_get_network_from_point():
#     point = AnnPoint(neuronCounts=[2, 4, 4, 3], actFuns=[ReLu(), ReLu(), Softmax()], lossFun=QuadDiff(), learningRate=1, momCoeff=2, batchSize=3)
#     network = network_from_point(point, 1010)
#
#     assert len(network.neuronCounts) == 4
#     assert network.neuronCounts[0] == 2
#     assert network.neuronCounts[1] == 4
#     assert network.neuronCounts[2] == 4
#     assert network.neuronCounts[3] == 3
#
#     assert len(network.actFuns) == 4
#     assert network.actFuns[0] is None
#     assert isinstance(network.actFuns[1], ReLu)
#     assert isinstance(network.actFuns[2], ReLu)
#     assert isinstance(network.actFuns[3], Softmax)
#
#     assert isinstance(network.lossFun, QuadDiff)
#     assert network.learningRate == 10
#     assert network.momCoeff == 100
#     assert network.batchSize == 8
#
#     assert network.layerCount == 4
#
#     assert len(network.weights) == 4
#     assert np.array_equal(network.weights[0], np.empty((0, 0)))
#     assert np.all(np.isclose(network.weights[1], np.array([[-0.83116718, -0.27092632], [-1.040413, -1.27319421],[0.09199489, 1.12827275], [0.70227065, -1.67139339]]), atol=1e-5))
#     assert np.all(np.isclose(network.weights[2], np.array([[-0.23979613, -0.82519097, -0.27174483,  0.38980573],
#                                                            [-0.25130439,  0.14294476,  1.35161869, -0.03725841],
#                                                           [-0.68505133,  0.17929361,  0.29902494, -0.1533996],
#                                                             [0.38148435, -0.4994389,   0.50463211, -0.12389273]]), atol=1e-5))
#     assert np.all(np.isclose(network.weights[3], np.array([[-1.2917587,   0.37861067, -0.88823027, -0.38984775],
#                                                            [-0.45926747, -0.19749436,  0.12400293,  0.14901888],
#                                                           [ 0.14190831, -0.23561145,  0.47601405, -0.31930159]]), atol=1e-5))
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

def test_numpy_deep_copy():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrixcp = matrix.copy()

    assert np.array_equal(matrix, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    matrix[2, 1] = 80
    matrix[1, 1] = 50
    matrix[1, 0] = 40

    assert np.array_equal(matrix, np.array([[1, 2, 3], [40, 50, 6], [7, 80, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    matrixcp[0, 1] = 20
    matrixcp[2, 0] = 70

    assert np.array_equal(matrix, np.array([[1, 2, 3], [40, 50, 6], [7, 80, 9]]))
    assert np.array_equal(matrixcp, np.array([[1, 20, 3], [4, 5, 6], [70, 8, 9]]))

def test_list_comparison():
    l1 = [0, 1, 2, 3]
    l2 = [0, 1, 2, 3]
    assert compare_lists(l1, l2)

    l2 = [0, 1, 2, 3, 4]
    assert not compare_lists(l1, l2)

    l2 = [2, 0, 1, 3]
    assert not compare_lists(l1, l2)

    l2 = []
    assert not compare_lists(l1, l2)

    l1 = []
    l2 = [0, 1, 2, 3]
    assert not compare_lists(l1, l2)

def test_wieght_mask():
    mask = get_weight_mask(1, 2, 7)

    assert np.array_equal(mask, np.array([[0, 1, 1, 1, 1, 0, 0],
                                          [0, 0, 1, 1, 1, 1, 1],
                                          [0, 1, 0, 1, 1, 1, 1],
                                          [0, 1, 1, 0, 1, 1, 1],
                                          [0, 1, 1, 1, 0, 1, 1],
                                          [0, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0]]))
def test_wieght_mask_2():
    mask = get_weight_mask(2, 1, 6)

    assert np.array_equal(mask, np.array([[0, 0, 1, 1, 1, 0],
                                          [0, 0, 1, 1, 1, 0],
                                          [0, 0, 0, 1, 1, 1],
                                          [0, 0, 1, 0, 1, 1],
                                          [0, 0, 1, 1, 0, 1],
                                          [0, 0, 0, 0, 0, 0]]))
def test_wieght_mask_3():
    mask = get_weight_mask(2, 2, 4)

    assert np.array_equal(mask, np.array([[0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0],
                                          [0, 0, 0, 0]]))

# test_wieght_mask()






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

# test_generate_population_limits()

# test_generate_population_limits()

# test_list_comparison()
# test_generate_population_limits()
# test_generate_population_limits()

def test_acts_same():
    acts1 = [ReLu(), TanH(), Sigmoid()]
    acts2 = [ReLu(), TanH(), Sigmoid()]
    acts3 = [ReLu(), TanH()]
    acts4 = [ReLu(), TanH(), SincAct()]

    assert_acts_same(acts1, acts2)

    try:
        assert_acts_same(acts1, acts3)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        assert_acts_same(acts4, acts1)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        assert_acts_same(acts2, acts4)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        assert_acts_same(acts4, acts3)
    except AssertionError:
        assert True
    else:
        assert False

    acts5 = [ReLu(), None, None]
    acts6 = [ReLu(), None]
    acts7 = [ReLu(), None, None]

    assert_acts_same(acts5, acts7)

    try:
        assert_acts_same(acts1, acts5)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        assert_acts_same(acts5, acts6)
    except AssertionError:
        assert True
    else:
        assert False

    try:
        assert_acts_same(acts1, acts6)
    except AssertionError:
        assert True
    else:
        assert False

# test_acts_same()
# test_copy_list_of_arrays()

test_generate_population_limits_aggr()