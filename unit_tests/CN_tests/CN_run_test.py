import numpy as np

from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
from utility.TestingUtility import assert_chaos_network_properties
from utility.Utility import *

#TODO - C - CN constructor

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
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)

    result = net.run(np.array([[0], [0]]))

    assert np.all(np.isclose(result, np.array([[0.416043846], [0.583956154]]), atol=1e-5))

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
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4],
                                    desired_inp=np.array([[0, 0, 0.5, 0.5, 0.744918662, 0.122459331, 0.461494578]]).reshape(-1, 1),
                                    desired_act=np.array([[0, 0, 0.622459331, 0.622459331, 0.678070495, 0.416043846, 0.583956154]]).reshape(-1, 1))

    net2 = net.copy()
    result2 = net2.run(np.array([[0], [0]]))

    assert np.all(np.isclose(result2, np.array([[0.416043846], [0.583956154]]), atol=1e-5))
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
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4],
                                    desired_inp=np.array([[0, 0, 0.5, 0.5, 0.744918662, 0.122459331, 0.461494578]]).reshape(-1, 1),
                                    desired_act=np.array([[0, 0, 0.622459331, 0.622459331, 0.678070495, 0.416043846, 0.583956154]]).reshape(-1, 1))

def test_cn_run2():
    weights = np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , -2, 1  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 2   , 0  ],
                        [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                        [0, 0, -2 , -1, 0  , -0.5, 0.5],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ],
                        [0, 0, 0  , 0 , 0  , 0   , 0  ]])

    links =   np.array([[0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)

    result = net.run(np.array([[0], [1]]))

    assert np.all(np.isclose(result, np.array([[0.504823439], [0.495176561]]), atol=1e-5))

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
                                                  [0, 0, 1, 0, 1, 0, 1],
                                                  [0, 0, 1, 1, 0, 1, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                                                    [0, 0, 0  , -2, 1  , 0   , 0  ],
                                                    [0, 0, 0  , 0 , 0  , 2   , 0  ],
                                                    [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                                                    [0, 0, -2 , -1, 0  , -0.5, 0.5],
                                                    [0, 0, 0  , 0 , 0  , 0   , 0  ],
                                                    [0, 0, 0  , 0 , 0  , 0   , 0  ]]),
                                    desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4],
                                    desired_inp=np.array([[0, 0, -0.604861199, -2.143643361, 0.552463361, -0.110888826, -0.130183178]]).reshape(-1, 1),
                                    desired_act=np.array([[0, 1, 0.353232317, 0.104926723, 0.634706921, 0.504823439, 0.495176561]]).reshape(-1, 1))

    net2 = net.copy()
    result2 = net2.run(np.array([[0], [1]]))

    assert np.all(np.isclose(result2, np.array([[0.504823439], [0.495176561]]), atol=1e-5))
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
                                                  [0, 0, 1, 0, 1, 0, 1],
                                                  [0, 0, 1, 1, 0, 1, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.5, 0 , 0  , 0   , 0  ],
                                                    [0, 0, 0  , -2, 1  , 0   , 0  ],
                                                    [0, 0, 0  , 0 , 0  , 2   , 0  ],
                                                    [0, 0, 1  , 0 , 0.5, 0   , 0.5],
                                                    [0, 0, -2 , -1, 0  , -0.5, 0.5],
                                                    [0, 0, 0  , 0 , 0  , 0   , 0  ],
                                                    [0, 0, 0  , 0 , 0  , 0   , 0  ]]),
                                    desired_biases=np.array([[0, 0, 0.5, 0.5, -0.5, -0.5, -0.5]]),
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4],
                                    desired_inp=np.array([[0, 0, -0.604861199, -2.143643361, 0.552463361, -0.110888826, -0.130183178]]).reshape(-1, 1),
                                    desired_act=np.array([[0, 1, 0.353232317, 0.104926723, 0.634706921, 0.504823439, 0.495176561]]).reshape(-1, 1))





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
    net = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                   net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5,
                   p_rad=-6)

    results = []

    for i in range(1000):
        results.append(net.run(np.array([[0.5], [-0.5]])))

    for i in range(999):
        assert np.array_equal(results[i], results[i + 1])

#TODO - C - test consecutive runs more

def test_run_with_cycle_1_run():
    links = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    weights = np.array([[0, 0, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, -1, 0],
                      [0, 0, 0, 1, 0, -1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    biases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    actFun = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    cn = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=biases, actFuns=actFun, aggrFun=Softmax(), net_it=1,
                  mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)

    res1 = cn.run(np.array([[-1, 0.5], [1, -2]]))

    assert np.all(np.isclose(res1, np.array([[0.268941421, 0.2566384], [0.731058579, 0.7433616]])))

    assert_chaos_network_properties(net=cn,
                                    desired_input_size=2,
                                    desired_output_size=2,
                                    desired_neuron_count=9,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=7,
                                    desired_hidden_count=5,
                                    desired_links=np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 1, 0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, -1, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, -1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, -1, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, -1, 0],
                                                    [0, 0, 0, 1, 0, -1, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=1,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 4, 5, 6, 3],
                                    desired_inp=np.array([[0, 0], [0, 0], [1, -0.5], [1, 1.063514312], [1, -2], [0.731058579, 0.377540669],
                                                [-0.731058579, -0.119202922], [-0.675037527, -0.593279805], [0.324962473, 0.470234507]]),
                                    desired_act=np.array([[-1, 0.5], [1, -2], [0.731058579, 0.377540669], [0.731058579, 0.743361562],
                                                [0.731058579, 0.119202922], [0.675037527, 0.593279805], [0.324962473, 0.470234507],
                                                [0.268941421, 0.256638438], [0.731058579, 0.743361562]]))

def test_run_with_cycle_2_run():
    links = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    weights = np.array([[0, 0, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, -1, 0],
                        [0, 0, 0, 1, 0, -1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    biases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

    actFun = [None, None, Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), None, None]
    cn = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=biases, actFuns=actFun,
                  aggrFun=Softmax(), net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4,
                  c_prob=-5, p_rad=-6)

    res1 = cn.run(np.array([[-1, 0.5], [1, -2]]))

    assert np.all(np.isclose(res1, np.array([[0.2764541, 0.2765315], [0.7235459, 0.7234685]])))

    assert_chaos_network_properties(net=cn,
                                    desired_input_size=2,
                                    desired_output_size=2,
                                    desired_neuron_count=9,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=7,
                                    desired_hidden_count=5,
                                    desired_links=np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 1, 0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, -1, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, -1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, -1, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, -1, 0],
                                                    [0, 0, 0, 1, 0, -1, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), Sigmoid(), None, None],
                                    desired_aggr=Softmax(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 4, 5, 6, 3],
                                    desired_inp=np.array([[0, 0],
                                                [0, 0],
                                                [1, -0.5],
                                                [0.962119343, 0.961732057],
                                                [0.268941421, -2.743361562],
                                                [0.406096106, -0.092693838],
                                                [-0.566833007, -0.060462661],
                                                [-0.60015143, -0.476843119],
                                                [0.361967913, 0.484888938]]),
                                    desired_act=np.array([[-1, 0.5],
                                                [1, -2],
                                                [0.731058579, 0.377540669],
                                                [0.723545932, 0.723468458],
                                                [0.566833007, 0.060462661],
                                                [0.60015143, 0.476843119],
                                                [0.361967913, 0.484888938],
                                                [0.276454068, 0.276531542],
                                                [0.723545932, 0.723468458]]))

# def test_staggered_run():
#     links = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 1, 0, 0, 0],
#                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
#                       [0, 0, 0, 1, 0, 0, 0, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 0, 0, 1],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])
#     weights = np.array([[0, 0, -1, 0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
#                         [0, 0, 0, 0, -1, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0, -1, 0, 0],
#                         [0, 0, 0, 1, 0, 0, 0, -1, 0],
#                         [0, 0, 0, 1, 0, -1, 0, 0, 1],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
#     biases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
#     cn = ChaosNet(input_size=2, output_size=2, links=links, weights=weights, biases=biases, actFuns=9 * [Sigmoid()], aggrFun=Softmax(), maxit=1)
#
#     res1 = cn.run(np.array([[-1], [1]]))
#     # assert np.all(np.isclose(cn.inp, np.array([[0, 0, 1, 1, 1, 0.731058579, -0.731058579, -0.675037527, 0.324962473]])))
#     # assert np.all(np.isclose(cn.act, np.array([[-1, 1, 0.731058579, 0.731058579, 0.731058579, 0.675037527, 0.324962473, 0.268941421, 0.731058579]])))
#     assert np.all(np.isclose(res1, np.array([[0.268941421], [0.731058579]])))
#
#     res2 = cn.run(np.array([[-1], [1]]))
    # assert np.all(np.isclose(cn.inp, np.array([[0, 0, 1, 0.9621193, 0.2689414, 0.4060961, -0.5668330, -0.6001514, 0.3619679]])))
    # assert np.all(np.isclose(cn.act, np.array([[-1, 1, 0.7310586, 0.7235459, 0.5668330, 0.6001514, 0.3619679, 0.2764541, 0.7235459]])))
    # assert np.all(np.isclose(res1, np.array([[0.2764541], [0.7235459]])))



def test_faster_run():
    weights = np.array([[0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 2, 1, 0],
                        [0, 0, 0, 0, -1, 0],
                        [0, 0, 0, 0, 0, -1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]])

    links =   np.array([[0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0]])

    bias = np.array([[0, 0, 0, 0, 1, -0.5]])
    actFuns = [None, None, None, ReLu(), ReLu(), None]
    net = ChaosNet(input_size=3, output_size=1, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Sigmoid(),
                   net_it=1, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5,
                   p_rad=-6)

    inputs = np.array([[0, 1], [1, 0], [0, 1]])
    expected = np.array([[0.377540669, 0.182425524]])
    res = net.run(inputs)

    assert np.all(np.isclose(res, expected))

    assert_chaos_network_properties(net=net,
                                    desired_input_size=3,
                                    desired_output_size=1,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=3,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0, 0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 1, 0],
                                                  [0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 2, 1, 0],
                                                    [0, 0, 0, 0, -1, 0],
                                                    [0, 0, 0, 0, 0, -1],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 1, -0.5]]),
                                    desired_actFun=[None, None, None, ReLu(), ReLu(), None],
                                    desired_aggr=Sigmoid(),
                                    desired_maxit=1,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[3, 4],
                                    desired_inp=np.array([[0, 0], [0, 0], [0, 0], [2, 1], [2, 0], [-0.5, -1.5]]),
                                    desired_act=np.array([[0, 1], [1, 0], [0, 1], [2, 1], [2, 0], [0.377540669, 0.182425524]]))

    o1 = net.run(inputs[:, 0].reshape((-1, 1)))
    o2 = net.run(inputs[:, 1].reshape((-1, 1)))

    assert np.all(np.isclose(o1, expected[0, 0], atol=1e-5))
    assert np.all(np.isclose(o2, expected[0, 1], atol=1e-5))

# test_cn_run()
# test_run_with_cycle_1_run()
# test_run_with_cycle_2_run()
# test_faster_run()
# test_faster_run()

# test_cn_run2()
