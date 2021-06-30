from utility.TestingUtility import compare_chaos_network
from utility.Utility import *

#TODO is CN constructor tested?

def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), (0, 3), [ReLu(), GaussAct(), Sigmoid()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6))
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1)


    assert len(nets) == 2

    compare_chaos_network(net=nets[0],
                          desired_input_size=2,
                          desited_output_size=1,
                          desired_neuron_count=3,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=2,
                          desired_hidden_count=0,
                          desired_links=np.array([[0, 0, 1],
                                                  [0, 0, 1],
                                                  [0, 0, 0]]),
                          desired_weights=np.array([[0, 0, -0.91773893],
                                                    [0, 0, -0.78179623],
                                                    [0, 0, 0]]),
                          desired_biases=np.array([[0, 0, 4.10972096]]),
                          desired_actFun=[None, None, None],
                          desired_aggr=ReLu(),
                          desired_maxit=1,
                          desired_mut_rad=0.555594,
                          desired_wb_prob=0.068865,
                          desired_s_prob=0.616022,
                          desired_p_prob=0.544396)

    compare_chaos_network(net=nets[1],
                          desired_input_size=2,
                          desited_output_size=1,
                          desired_neuron_count=6,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=5,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 1, 1, 0, 1],
                                                  [0, 0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 1, 1],
                                                  [0, 0, 1, 0, 1, 1],
                                                  [0, 0, 1, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0.02348412, -0.47971685, 0, 0.0128531 ],
                                                    [0, 0, 0, -0.78054184, 0.46013031, 0],
                                                    [0, 0, 0, -0.91486681 , -0.11334042, 0.78668504],
                                                    [0, 0, 0.32327473, 0, 0.39699364, 0.2487557],
                                                    [0, 0, 0.5869129, 0.96277274, 0, -0.68333513],
                                                    [0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, 0, -5.97746074, 5.44356359, 0.04376756, 5.26342884]]),
                          desired_actFun=[None, None, ReLu(), Sigmoid(), ReLu(), None],
                          desired_aggr=ReLu(),
                          desired_maxit=4,
                          desired_mut_rad=0.714939,
                          desired_wb_prob=0.082788,
                          desired_s_prob=0.673710,
                          desired_p_prob=0.437508)

random.seed(1001)
np.random.seed(1001)

hd1 = random.randint(0, 3)
print(f"hd1: {hd1}")
n1 = 2 + 1 + hd1
print(f"den1: {random.random()}")
print(f"lin_prob1: \n{np.random.random((n1, n1))}")
print(f"wei_1: \n{np.random.uniform(-1, 1, (n1, n1))}")
print(f"bia_1: \n{np.random.uniform(-10, 10, (1, n1))}")
print(f"aggrf_1: {random.randint(0, 2)}")
print(f"maxit_1: {random.randint(1, 5)}")
print(f"mut_rad_1: {random.uniform(0, 1)}")
print(f"wb_prob_1: {random.uniform(0.05, 0.1)}")
print(f"s_prob_1: {random.uniform(0.6, 0.7)}")
print(f"p_prob_1: {random.uniform(0.4, 0.6)}")

print()

hd2 = random.randint(0, 3)
print(f"hd2: {hd2}")
n2 = 2 + 1 + hd2
print(f"den2: {random.random()}")
print(f"lin_prob2: \n{np.random.random((n2, n2))}")
print(f"wei_2: \n{np.random.uniform(-1, 1, (n2, n2))}")
print(f"bia_2: \n{np.random.uniform(-10, 10, (1, n2))}")
print(f"af2_2: {random.randint(0, 2)}")
print(f"af2_3: {random.randint(0, 2)}")
print(f"af2_4: {random.randint(0, 2)}")
print(f"aggrf_2: {random.randint(0, 2)}")
print(f"maxit_2: {random.randint(1, 5)}")
print(f"mut_rad_2: {random.uniform(0, 1)}")
print(f"wb_prob_2: {random.uniform(0.05, 0.1)}")
print(f"s_prob_2: {random.uniform(0.6, 0.7)}")
print(f"p_prob_2: {random.uniform(0.4, 0.6)}")


# test_cn_generation()