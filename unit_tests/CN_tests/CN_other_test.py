from math import ceil

from sklearn import datasets
import numpy as np
import pytest

from ann_point.Functions import *
from neural_network.LsmNetwork import LsmNetwork, efficiency
from utility.TestingUtility import assert_chaos_network_properties
import random

from utility.Utility import generate_counting_problem, one_hot_endode
from utility.Utility2 import get_weight_mask

def test_CN_constr():
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
    actFuns = [None, None, Sigmoid(), TanH(), ReLu(), None, None]
    net = LsmNetwork(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                     net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4,
                     c_prob=-5, p_rad=-6)

    assert_chaos_network_properties(net=net,
                                    desired_input_size=2,
                                    desired_output_size=2,
                                    desired_neuron_count=7,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                                    desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=None,
                                    desired_inp=np.zeros((0,0)),
                                    desired_act=np.zeros((0,0)))


def test_CN_copy():
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
    actFuns = [None, None, Sigmoid(), TanH(), ReLu(), None, None]
    net = LsmNetwork(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                     net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4,
                     c_prob=-5, p_rad=-6)


    net.run(np.array([[0], [1]]))
    net2 = net.copy()

    assert_chaos_network_properties(net=net,
                                    desired_input_size=2,
                                    desired_output_size=2,
                                    desired_neuron_count=7,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                                    desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4],
                                    desired_inp=np.array([[0, 0, 0.5, -0.5, -0.924234315, 0.122459331, -0.962117157]]).reshape(-1, 1),
                                    desired_act=np.array([[0, 1, 0.622459331, -0.462117157, 0, 0.747359064, 0.252640936]]).reshape(-1, 1))

    net.net_it = 32
    net.links[-1, -3] = 32313
    net.actFuns[2] = Identity()


    assert_chaos_network_properties(net=net2,
                                    desired_input_size=2,
                                    desired_output_size=2,
                                    desired_neuron_count=7,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.5, 0, 0, 0, 0],
                                                    [0, 0, 0, -1, 0.5, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 0, 2, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0.5],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                                    desired_actFun=[None, None, Sigmoid(), TanH(), ReLu(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6)

def test_edge_count():
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
    actFuns = [None, None, Sigmoid(), TanH(), ReLu(), None, None]
    net = LsmNetwork(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                     net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4,
                     c_prob=-5, p_rad=-6)

    assert net.get_edge_count() == 7


def test_cn_for_prob():
    seed = 22223333
    random.seed(seed)
    np.random.seed(seed)

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x = [x.reshape((4, 1)) for x in x]
    y = one_hot_endode(y)
    perm = list(range(0, len(y)))
    random.shuffle(perm)

    count_tr = 500
    count_test = 500
    size = 3
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)

    links = np.array([[0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 1., 1., 0., 1., 1.],
                      [0., 0., 0., 1., 0., 0., 1., 1., 0., 1.],
                      [0., 0., 0., 0., 1., 0., 0., 1., 0., 1.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    wei = np.array([[ 0.         , 0.         , 0.        ,  0.02870241 , 0.         ,-0.65170473, 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        ,  0.         ,-0.50803933 ,-0.51046493, 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        , -0.59235079 , 0.         , 0.        , 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        ,  0.         , 0.         ,-0.26603819, -0.47900631  ,0.          ,-0.7692094  ,-0.84285979],
                    [-0.         , 0.         , 0.        , -0.3525011  , 0.         , 0.        , 0.00727213   ,-0.17702534 , 0.         ,-0.81879874],
                    [ 0.         , 0.         , 0.        ,  0.         ,-0.39768254 , 0.        , 0.           ,-0.14567781 , 0.         ,-0.80649623],
                    [ 0.         , 0.         , 0.        ,  0.         , 0.         , 0.        , 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        ,  0.         , 0.         , 0.        , 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        ,  0.         , 0.         , 0.        , 0.           ,0.          , 0.         , 0.        ],
                    [ 0.         , 0.         , 0.        ,  0.         , 0.         , 0.        , 0.           ,0.          , 0.         , 0.        ]])

    biases = np.array([[-0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -6.74547958e-01, -1.01664251e+00, -1.03673969e+00,  3.35911962e-01,  6.39615192e-01,-2.68632982e-02,  6.13634466e-05]])

    acts = [None, None, None, Poly3(), GaussAct(), GaussAct(), None, None, None, None]

    net = LsmNetwork(input_size=3, output_size=4, links=links, weights=wei, biases=biases, actFuns=acts,
                     aggrFun=Sigmoid(), net_it=6, mutation_radius=-1, swap_prob=-2, multi=-3,
                     p_prob=-4, c_prob=-5, p_rad=-6)
    test_res = net.test(X, Y)

    assert efficiency(test_res[0]) == pytest.approx(0.8367295847418645, abs=1e-6)

