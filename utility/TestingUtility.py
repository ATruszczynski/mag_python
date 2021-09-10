from math import log10

from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from neural_network.LsmNetwork import LsmNetwork, ActFun
import numpy as np
import pytest

def assert_chaos_network_properties(net: LsmNetwork,
                                    desired_input_size: int,
                                    desired_output_size: int,
                                    desired_neuron_count: int,
                                    desired_hidden_start_index: int,
                                    desired_hidden_end_index: int,
                                    desired_hidden_count: int,
                                    desired_links: np.ndarray,
                                    desired_weights: np.ndarray,
                                    desired_biases: np.ndarray,
                                    desired_actFun: [ActFun],
                                    desired_aggr: ActFun, desired_maxit: int,
                                    desired_mut_rad: float,
                                    desired_wb_prob: float,
                                    desired_s_prob: float,
                                    desired_p_prob: float,
                                    desired_c_prob: float,
                                    desired_r_prob: float,
                                    desired_hidden_comp_order: [int] = None,
                                    desired_inp: np.ndarray = None,
                                    desired_act: np.ndarray = None):
    assert net.input_size == desired_input_size
    assert net.output_size == desired_output_size
    assert net.neuron_count == desired_neuron_count
    assert net.hidden_start_index == desired_hidden_start_index
    assert net.hidden_end_index == desired_hidden_end_index
    assert net.hidden_count == desired_hidden_count
    if desired_inp is None:
        desired_inp = np.zeros((0, 0))
    if desired_act is None:
        desired_act = np.zeros((0, 0))
    assert net.inp.shape == desired_inp.shape and np.all(np.isclose(net.inp, desired_inp, atol=1e-5))
    assert net.act.shape == desired_act.shape and np.all(np.isclose(net.act, desired_act, atol=1e-5))

    assert net.links.shape == desired_links.shape and np.array_equal(net.links, desired_links)
    assert net.weights.shape == desired_weights.shape and np.all(np.isclose(net.weights, desired_weights, atol=1e-5))
    assert net.biases.shape == desired_biases.shape and np.all(np.isclose(net.biases, desired_biases, atol=1e-5))
    assert len(net.actFuns) == len(desired_actFun)
    for i in range(len(net.actFuns)):
        assert net.actFuns[i] is None or net.actFuns[i].to_string() == desired_actFun[i].to_string()
    assert net.aggrFun.to_string() == desired_aggr.to_string()

    if desired_hidden_comp_order is None or net.hidden_comp_order is None:
        assert desired_hidden_comp_order is None and net.hidden_comp_order is None
    else:
        assert len(net.hidden_comp_order) == len(desired_hidden_comp_order)
        for i in range(len(net.hidden_comp_order)):
            assert net.hidden_comp_order[i] == desired_hidden_comp_order[i]

    assert net.net_it == desired_maxit
    assert net.mutation_radius == pytest.approx(desired_mut_rad, 1e-4)
    assert net.swap_prob == pytest.approx(desired_wb_prob, 1e-4)
    assert net.multi == pytest.approx(desired_s_prob, 1e-4)
    assert net.p_prob == pytest.approx(desired_p_prob, 1e-4)
    assert net.c_prob == pytest.approx(desired_c_prob, 1e-4)
    assert net.p_rad == pytest.approx(desired_r_prob, 1e-4)

def assert_chaos_networks_same(net: LsmNetwork, net2: LsmNetwork):
    assert_chaos_network_properties(net,
                                    desired_input_size=net2.input_size,
                                    desired_output_size=net2.output_size,
                                    desired_neuron_count=net2.neuron_count,
                                    desired_hidden_start_index=net2.hidden_start_index,
                                    desired_hidden_end_index=net2.hidden_end_index,
                                    desired_hidden_count=net2.hidden_count,
                                    desired_links=net2.links,
                                    desired_weights=net2.weights,
                                    desired_biases=net2.biases,
                                    desired_actFun=net2.actFuns,
                                    desired_aggr=net2.aggrFun,
                                    desired_maxit=net2.net_it,
                                    desired_mut_rad=net2.mutation_radius,
                                    desired_wb_prob=net2.swap_prob,
                                    desired_s_prob=net2.multi,
                                    desired_p_prob=net2.p_prob,
                                    desired_c_prob=net2.c_prob,
                                    desired_r_prob=net2.p_rad,
                                    desired_hidden_comp_order=net2.hidden_comp_order,
                                    desired_inp=net2.inp,
                                    desired_act=net2.act)


