from utility.TestingUtility import assert_chaos_network_properties
from utility.Utility import *

def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), (0, 3), [ReLu(), GaussAct(), Sigmoid()], mut_radius=(-1, 0),
                                 swap=(-2, -1), multi=(-3, -2), p_prob=(-4, -3), c_prob=(-5, -4),
                                 p_rad=(-6, -5))
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1)

    assert len(nets) == 2

    assert_chaos_network_properties(net=nets[0],
                                    desired_input_size=2,
                                    desired_output_size=1,
                                    desired_neuron_count=3,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=2,
                                    desired_hidden_count=0,
                                    desired_links=np.array([[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0],
                                                    [0, 0, 0],
                                                    [0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 4.10972096]]),
                                    desired_actFun=[None, None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.4444052,
                                    desired_wb_prob=-1.62269240,
                                    desired_s_prob=-2.8397767,
                                    desired_p_prob=-3.27801995,
                                    desired_c_prob=-4.43578206,
                                    desired_r_prob=-5.22632503)

    assert_chaos_network_properties(net=nets[1],
                                    desired_input_size=2,
                                    desired_output_size=1,
                                    desired_neuron_count=4,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=1,
                                    desired_links=np.array([[0, 0, 1, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.75720675, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 5.2041659, 4.66856333]]),
                                    desired_actFun=[None, None, ReLu(), None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=4,
                                    desired_mut_rad=-0.28506096,
                                    desired_wb_prob=-1.34422750,
                                    desired_s_prob=-2.2628928,
                                    desired_p_prob=-3.812457778,
                                    desired_c_prob=-4.498022163,
                                    desired_r_prob=-5.1049701)


def test_cn_generation_aggr():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), (0, 3), [ReLu(), GaussAct(), Sigmoid()], mut_radius=(-1, 0),
                                 swap=(-2, -1), multi=(-3, -2), p_prob=(-4, -3), c_prob=(-5, -4),
                                 p_rad=(-6, -5), aggrFuns=[Poly2(), Identity(), Poly3(), SincAct()])

    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1)

    assert len(nets) == 2

    assert_chaos_network_properties(net=nets[0],
                                    desired_input_size=2,
                                    desired_output_size=1,
                                    desired_neuron_count=3,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=2,
                                    desired_hidden_count=0,
                                    desired_links=np.array([[0, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0],
                                                              [0, 0, 0],
                                                              [0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 4.10972096]]),
                                    desired_actFun=[None, None, None],
                                    desired_aggr=Identity(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.4444052,
                                    desired_wb_prob=-1.62269240,
                                    desired_s_prob=-2.8397767,
                                    desired_p_prob=-3.27801995,
                                    desired_c_prob=-4.43578206,
                                    desired_r_prob=-5.22632503)

    assert_chaos_network_properties(net=nets[1],
                                    desired_input_size=2,
                                    desired_output_size=1,
                                    desired_neuron_count=4,
                                    desired_hidden_start_index=2,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=1,
                                    desired_links=np.array([[0, 0, 1, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 0, 0.75720675, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, 0, 5.2041659, 4.66856333]]),
                                    desired_actFun=[None, None, ReLu(), None],
                                    desired_aggr=Poly2(),
                                    desired_maxit=4,
                                    desired_mut_rad=-0.28506096,
                                    desired_wb_prob=-1.34422750,
                                    desired_s_prob=-2.2628928,
                                    desired_p_prob=-3.812457778,
                                    desired_c_prob=-4.498022163,
                                    desired_r_prob=-5.1049701)
