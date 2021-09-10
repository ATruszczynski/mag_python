import pytest

from utility.Mut_Utility import *
from utility.TestingUtility import assert_chaos_network_properties
from utility.Utility import *

def test_neuron_increase():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 p_rad=(0.44, 0.55))

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0, 0, 3, 8, 5],
                     [0, 7, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), Sigmoid(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                     actFuns=actFuns1, aggrFun=TanH(), net_it=2,
                     mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)

    np.random.seed(1001)
    random.seed(1001)


    cn2 = change_neuron_count(cn1, hrange, 4)


    ############################################################################

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0., 1, 2, 0, 0],
                                                    [0, 0, 3, 8, 5],
                                                    [0, 7, 0, 0, 6],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5]]),
                                    desired_actFun=[None, ReLu(), Sigmoid(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6)

    assert_chaos_network_properties(net=cn2,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=7,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=4,
                                    desired_links=np.array([[0., 1., 1., 1., 1., 0., 0.],
                                                  [0., 0., 1., 0., 1., 1., 1.],
                                                  [0., 1., 0., 1., 0., 0., 1.],
                                                  [0., 1., 1., 0., 0., 1., 1.],
                                                  [0., 0., 1., 1., 0., 1., 1.],
                                                  [0., 0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 1, 2, 1.99852202, 3.53627653, 0, 0],
                                                    [0 , 0, 3, 0, 2.08113262, 8, 5],
                                                    [0 , 7, 0, 0.87783263, 0, 0, 6],
                                                    [0., 7.15815828, 0.34053276, 0., 0., 5.35134103, 6.56899812],
                                                    [0., 0., 5.58797454, 4.99502278, 0., 7.54306312, 6.34765188],
                                                    [0 , 0, 0, 0, 0, 0, 0],
                                                    [0 , 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -0.54751109, -1.92308629, -4, -5]]),
                                    desired_actFun=[None, ReLu(), Sigmoid(), SincAct(), ReLu(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_inp=np.zeros((0,0)),
                                    desired_act=np.zeros((0,0)))

def test_neuron_decrease():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 p_rad=(0.44, 0.55))

    link1 = np.array([[0, 1, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1],
                      [0, 1, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 =  np.array([[0, 1, 4, 0, 0, 0],
                      [0, 0, 5, 7, 9, 11],
                      [0, 2, 0, 8, 0, 12],
                      [0, 3, 6, 0, 0, 13],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), Sigmoid(), SincAct(), None, None]

    cn1 = LsmNetwork(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                     actFuns=actFuns1, aggrFun=TanH(), net_it=2,
                     mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = change_neuron_count(cn1, hrange, 1)

    ##########################################################################

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 1, 0, 0, 0],
                                                  [0, 0, 1, 1, 1, 1],
                                                  [0, 1, 0, 1, 0, 1],
                                                  [0, 1, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 1, 4, 0, 0, 0],
                                                    [0, 0, 5, 7, 9, 11],
                                                    [0, 2, 0, 8, 0, 12],
                                                    [0, 3, 6, 0, 0, 13],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0.]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), Sigmoid(), SincAct(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_inp=np.zeros((0,0)),
                                    desired_act=np.zeros((0,0)))

    ##########################################################################

    assert_chaos_network_properties(net=cn2,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=4,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=2,
                                    desired_hidden_count=1,
                                    desired_links=np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0, 0],
                                                    [0, 0, 0, 13],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0.]]),
                                    desired_biases=np.array([[0., -4, -5, -6]]),
                                    desired_actFun=[None, SincAct(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_inp=np.zeros((0,0)),
                                    desired_act=np.zeros((0,0)))


def test_gaussian_shift():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = gaussian_shift(weights, links, 0.75, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1.50308048, 1.96986593],
                                                [0, 0, -0.90477989],
                                                [0, 0, 0]]), atol=1e-4))


    seed = 1004
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = gaussian_shift(weights, links, 0.75, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1, 5.72960772],
                                                [0, 0, 6.10237258],
                                                [0, 0, 0]]), atol=1e-4))




def test_conditional_try_differnt():
    set = [ReLu(), Poly2(), SincAct(), Poly3()]

    random.seed(1001)

    f1 = conditional_try_choose_different(0.5, ReLu(), set)
    assert f1.to_string() == "RL"

    f1 = conditional_try_choose_different(0.5, ReLu(), set)
    assert f1.to_string() == "P2"

    f1 = conditional_try_choose_different(0.5, SincAct(), set)
    assert f1.to_string() == "SC"

    f1 = conditional_try_choose_different(0.5, Poly3(), set)
    assert f1.to_string() == "P2"


def test_conditional_value_swap():
    random.seed(1006)

    a = 10
    b = 20
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 20
    assert d == 10

    a = 20
    b = 30
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 30
    assert d == 20

    a = 30
    b = 40
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 40
    assert d == 30

    a = 40
    b = 50
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 40
    assert d == 50

def test_uniform_shift():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = uniform_shift(weights, links, 0.6, -2, 1)

    assert np.all(np.isclose(weights, np.array([[0, 1.56070304, 0.12339161],
                                                [0, 0, 2.32730565],
                                                [0, 0, 0]]), atol=1e-4))


    seed = 1004
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = uniform_shift(weights, links, 0.6, -2, 1)

    assert np.all(np.isclose(weights, np.array([[0, 1, 2],
                                                [0, 0, 3.31546711],
                                                [0, 0, 0]]), atol=1e-4))


def test_add_remove_weights():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 4), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 p_rad=(0.44, 0.55))


    link1 = np.array([[0., 1, 0, 0, 0],
                      [0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])

    wei1 =  np.array([[0, 1, 0, 0, 0],
                      [0, 0, 3, 5, 7],
                      [0, 2, 4, 0, 8],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0.]])

    seed = 11002211
    random.seed(seed)
    np.random.seed(seed)
    link2, wei2 = add_or_remove_edges(0.8, link1, wei1, get_weight_mask(1, 2, 5), hrange)

    explink = np.array([[0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])

    expwei = np.array([[0, 0, 2.5023659, 0, 0],
                       [0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])

    assert np.array_equal(explink, link2)
    assert np.all(np.isclose(expwei, wei2, atol=1e-5))

def test_add_remove_weights_2():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 4), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 p_rad=(0.44, 0.55))


    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., -1, 4, 0],
                      [0 , 0, -2, 5.5],
                      [0 , 0, 0, 3],
                      [0 , 0, 0, 0]])

    seed = 67859404
    random.seed(seed)
    np.random.seed(seed)
    link2, wei2 = add_or_remove_edges(0.8, link1, wei1, get_weight_mask(1, 1, 4), hrange)

    explink = np.array([[0., 0, 0, 0],
                        [0 , 0, 0, 0],
                        [0 , 1, 0, 1],
                        [0 , 0, 0, 0]])

    expwei =  np.array([[0., 0, 0, 0],
                        [0 , 0, 0, 0],
                        [0 , 0.89671571, 0, 3],
                        [0 , 0, 0, 0]])

    assert np.array_equal(explink, link2)
    assert np.all(np.isclose(expwei, wei2, atol=1e-5))

def test_add_remove_weights_3():
    hrange = HyperparameterRange(init_wei=(-1, 2), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 4), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 swap=(0.05, 0.1), multi=(0.6, 0.7), p_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 p_rad=(0.44, 0.55))


    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., 0, 0, 0],
                      [0 , 0, 0, 0],
                      [0 , 0, 0, 0],
                      [0 , 0, 0, 0]])

    seed = 67859404
    random.seed(seed)
    np.random.seed(seed)
    link2, wei2 = add_or_remove_edges(0.8, link1, wei1, get_weight_mask(1, 1, 4), hrange)

    explink = np.array([[0., 0, 0, 0],
                        [0 , 0, 0, 0],
                        [0 , 1, 0, 1],
                        [0 , 0, 0, 0]])

    expwei =  np.array([[0., 0, 0, 0],
                        [0 , 0, 0, 0],
                        [0 , 0.15868628, 0, 0],
                        [0 , 0, 0, 0]])

    assert np.array_equal(explink, link2)
    assert np.all(np.isclose(expwei, wei2, atol=1e-5))

def test_min_max_weights_from_matrix():
    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., 0, 0, 0],
                      [0 , 0, 0, 0],
                      [0 , 0, 0, 0],
                      [0 , 0, 0, 0]])

    minW, maxW = get_min_max_values_of_matrix_with_mask(wei1, link1)

    assert minW == 0
    assert maxW == 0


    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., 0, 0, 1],
                      [0 , 0, 0, 0],
                      [0 , 1, 0, 0],
                      [0 , 0, 0, 0]])

    minW, maxW = get_min_max_values_of_matrix_with_mask(wei1, link1)

    assert minW == 0
    assert maxW == 0


    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., 2, 3, 0],
                      [0 , 0, 3, 1],
                      [0 , 0, 0, 4],
                      [0 , 0, 0, 0]])

    minW, maxW = get_min_max_values_of_matrix_with_mask(wei1, link1)

    assert minW == 1
    assert maxW == 4


    link1 = np.array([[0., 1, 1, 0],
                      [0 , 0, 1, 1],
                      [0 , 0, 0, 1],
                      [0 , 0, 0, 0]])

    wei1 =  np.array([[0., 2, -3, 0],
                      [0 , 0, 0, 5],
                      [0 , 0, 0, 4],
                      [0 , 0, 0, 0]])

    minW, maxW = get_min_max_values_of_matrix_with_mask(wei1, link1)

    assert minW == -3
    assert maxW == 5

#TODO - C - ten plik jest za duży
def test_uniform_value_shift():
    random.seed(10012001)

    val1 = conditional_uniform_value_shift(0.8, 0.1, 0., 1, 0.1)
    assert val1 == pytest.approx(0.13521542, abs=1e-5)

    val1 = conditional_uniform_value_shift(0.8, 0.25, 0., 1, 0.15)
    assert val1 == pytest.approx(0.299193710, abs=1e-5)

    val1 = conditional_uniform_value_shift(0.6, 0.25, 0., 1, 0.15)
    assert val1 == pytest.approx(0.25, abs=1e-5)

    results = []
    for i in range(200):
        results.append(conditional_uniform_value_shift(1, 0.25, 0., 1, i * 0.05))

    assert min(results) >= 0.
    assert min(results) <= 0.05
    assert max(results) <= 1.
    assert max(results) >= 0.95

def test_gaussian_value_shift():
    random.seed(10012001)

    val1 = conditional_gaussian_value_shift(0.8, 0.1, 0., 1, 0.1)
    assert val1 == pytest.approx(0.0573297, abs=1e-5)

    val1 = conditional_gaussian_value_shift(0.8, 0.5, 0., 2, 0.15)
    assert val1 == pytest.approx(0.244504543, abs=1e-5)

    val1 = conditional_gaussian_value_shift(0.6, 0.25, 0., 1, 0.15)
    assert val1 == pytest.approx(0.25, abs=1e-5)

    results = []
    for i in range(200):
        results.append(conditional_gaussian_value_shift(1, 0.25, 0., 1, i * 0.05))

    assert min(results) == 0.
    assert min(results) <= 0.05
    assert max(results) == 1.
    assert max(results) >= 0.95



