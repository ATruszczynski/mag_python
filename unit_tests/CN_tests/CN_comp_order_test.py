from ann_point.Functions import ReLu
from neural_network.LsmNetwork import LsmNetwork
import numpy as np

from utility.TestingUtility import assert_chaos_network_properties


def test_cn_comp_order():
    links = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
    biases = np.array([[0, 0, 0, 0, 0, 0]])
    cn = LsmNetwork(input_size=2, output_size=1, links=links, weights=links, biases=biases, actFuns=6 * [None], aggrFun=ReLu(),
                    net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn.compute_comp_order()

    assert_chaos_network_properties(net=cn,
                                    desired_input_size=2,
                                    desired_output_size=1,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 0, 0, 1, 0, 0],
                                                  [0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 1, 1, 0],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0, 1, 0, 0],
                                                    [0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 1, 1, 0],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, None, None, None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 3, 4])

def test_cn_comp_order_2():
    links = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
    biases = np.array([[0, 0, 0, 0, 0, 0]])
    cn = LsmNetwork(input_size=1, output_size=1, links=links, weights=links, biases=biases, actFuns=6 * [None], aggrFun=ReLu(),
                    net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn.compute_comp_order()

    assert_chaos_network_properties(net=cn,
                                    desired_input_size=1,
                                    desired_output_size=1,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=4,
                                    desired_links=np.array([[0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, None, None, None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[1])

def test_cn_comp_order_3():
    links = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0]])
    biases = np.array([[0, 0, 0, 0, 0, 0]])
    cn = LsmNetwork(input_size=1, output_size=1, links=links, weights=links, biases=biases, actFuns=6 * [None], aggrFun=ReLu(),
                    net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn.compute_comp_order()

    assert_chaos_network_properties(net=cn,
                                    desired_input_size=1,
                                    desired_output_size=1,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=5,
                                    desired_hidden_count=4,
                                    desired_links=np.array([[0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, None, None, None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[1])


def test_cn_comp_order_rec():
    links = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    biases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    cn = LsmNetwork(input_size=2, output_size=2, links=links, weights=links, biases=biases, actFuns=9 * [None], aggrFun=ReLu(),
                    net_it=2, mutation_radius=-1, swap_prob=-2, multi=-3, p_prob=-4, c_prob=-5, p_rad=-6)
    cn.compute_comp_order()


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
                                    desired_weights=np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 1, 0, 0, 0, 1, 0],
                                                    [0, 0, 0, 1, 0, 1, 0, 0, 1],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                    desired_actFun=[None, None, None, None, None, None, None, None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=2,
                                    desired_mut_rad=-1,
                                    desired_wb_prob=-2,
                                    desired_s_prob=-3,
                                    desired_p_prob=-4,
                                    desired_c_prob=-5,
                                    desired_r_prob=-6,
                                    desired_hidden_comp_order=[2, 4, 5, 6, 3])




