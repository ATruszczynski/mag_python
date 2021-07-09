from neural_network.ChaosNet import ChaosNet, ActFun
import numpy as np
import pytest

#TODO test this !!!
#TODO this may require even more asserts
def compare_chaos_network(net: ChaosNet,
                          desired_input_size: int,
                          desited_output_size: int,
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
    assert net.output_size == desited_output_size
    assert net.neuron_count == desired_neuron_count
    assert net.hidden_start_index == desired_hidden_start_index
    assert net.hidden_end_index == desired_hidden_end_index
    assert net.hidden_count == desired_hidden_count
    if desired_inp is None:
        desired_inp = np.zeros((1, desired_neuron_count))
    if desired_act is None:
        desired_act = np.zeros((1, desired_neuron_count))
    assert np.all(np.isclose(net.inp, desired_inp, atol=1e-5))
    assert np.all(np.isclose(net.act, desired_act, atol=1e-5))

    assert np.array_equal(net.links, desired_links)
    assert np.all(np.isclose(net.weights, desired_weights, atol=1e-5))
    assert np.all(np.isclose(net.biases, desired_biases, atol=1e-5))
    assert len(net.actFuns) == len(desired_actFun)
    for i in range(len(net.actFuns)):
        assert net.actFuns[i] is None or net.actFuns[i].to_string() == desired_actFun[i].to_string()
    assert net.aggrFun.to_string() == desired_aggr.to_string()

    if desired_hidden_comp_order is None:
        assert net.hidden_comp_order is None
    else:
        assert len(net.hidden_comp_order) == len(desired_hidden_comp_order)
        for i in range(len(net.hidden_comp_order)):
            assert net.hidden_comp_order[i] == desired_hidden_comp_order[i]

    assert net.maxit == desired_maxit
    assert net.mutation_radius == pytest.approx(desired_mut_rad, 1e-4)
    assert net.wb_mutation_prob == pytest.approx(desired_wb_prob, 1e-4)
    assert net.s_mutation_prob == pytest.approx(desired_s_prob, 1e-4)
    assert net.p_mutation_prob == pytest.approx(desired_p_prob, 1e-4)
    assert net.c_prob == pytest.approx(desired_c_prob, 1e-4)
    assert net.r_prob == pytest.approx(desired_r_prob, 1e-4)
