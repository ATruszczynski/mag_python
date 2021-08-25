from utility.TestingUtility import assert_chaos_network_properties
from utility.Utility import *

# TODO - C - clean up those files ffs
def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), (0, 3), [ReLu(), GaussAct(), Sigmoid()], mut_radius=(-1, 0),
                                 sqr_mut_prob=(-2, -1), lin_mut_prob=(-3, -2), p_mutation_prob=(-4, -3), c_prob=(-5, -4),
                                 dstr_mut_prob=(-6, -5))
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1)

    assert len(nets) == 2

    assert_chaos_network_properties(net=nets[0],
                                    desired_input_size=2,
                                    desited_output_size=1,
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
                                    desired_r_prob=-5.226325)

    assert_chaos_network_properties(net=nets[1],
                                    desired_input_size=2,
                                    desited_output_size=1,
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
                                    desired_mut_rad=-0.2850609,
                                    desired_wb_prob=-1.3442275,
                                    desired_s_prob=-2.2628928,
                                    desired_p_prob=-3.8124577,
                                    desired_c_prob=-4.498022,
                                    desired_r_prob=-5.1049701)

# random.seed(1001)
# np.random.seed(1001)
#
# hd1 = random.randint(0, 3)
# print(f"hd1: {hd1}")
# n1 = 2 + 1 + hd1
# print(f"den1: {random.random()}")
# print(f"lin_prob1: \n{np.random.random((n1, n1))}")
# print(f"wei_1: \n{np.random.uniform(-1, 1, (n1, n1))}")
# print(f"bia_1: \n{np.random.uniform(-10, 10, (1, n1))}")
# print(f"aggrf_1: {random.randint(0, 2)}")
# print(f"maxit_1: {random.randint(1, 5)}")
# print(f"mut_rad_1: {random.uniform(-1, 0)}")
# print(f"wb_prob_1: {random.uniform(-2, -1)}")
# print(f"s_prob_1: {random.uniform(-3, -2)}")
# print(f"p_prob_1: {random.uniform(-4, -3)}")
# print(f"c_prob_1: {random.uniform(-5, -4)}")
# print(f"r_prob_1: {random.uniform(-6, -5)}")
#
# print()
#
# hd2 = random.randint(0, 3)
# print(f"hd2: {hd2}")
# n2 = 2 + 1 + hd2
# print(f"den2: {random.random()}")
# print(f"lin_prob2: \n{np.random.random((n2, n2))}")
# print(f"wei_2: \n{np.random.uniform(-1, 1, (n2, n2))}")
# print(f"bia_2: \n{np.random.uniform(-10, 10, (1, n2))}")
# print(f"af2_2: {random.randint(0, 2)}")
# print(f"aggrf_2: {random.randint(0, 2)}")
# print(f"maxit_2: {random.randint(1, 5)}")
# print(f"mut_rad_2: {random.uniform(-1, 0)}")
# print(f"wb_prob_2: {random.uniform(-2, -1)}")
# print(f"s_prob_2: {random.uniform(-3, -2)}")
# print(f"p_prob_2: {random.uniform(-4, -3)}")
# print(f"c_prob_2: {random.uniform(-5, -4)}")
# print(f"r_prob_2: {random.uniform(-6, -5)}")


# test_cn_generation()