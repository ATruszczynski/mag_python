from utility.Utility import *

#TODO is CN constructor tested?

def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), (0, 3), [ReLu(), GaussAct(), Sigmoid()])
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1)


    assert len(nets) == 2
    assert np.array_equal(nets[0].links, np.array([[0, 0, 1],
                                                   [0, 0, 1],
                                                   [0, 0, 0]]))
    assert np.all(np.isclose(nets[0].weights, np.array([[0, 0, -0.91773893],
                                                        [0, 0, -0.78179623],
                                                        [0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(nets[0].biases, np.array([[0, 0, 4.10972096]]), atol=1e-5))
    assert len(nets[0].actFuns) == 3
    assert nets[0].actFuns[0] is None
    assert nets[0].actFuns[1] is None
    assert nets[0].actFuns[2] is None
    assert nets[0].aggrFun.to_string() == ReLu().to_string()
    assert nets[0].hidden_comp_order is None
    assert nets[0].maxit == 1

    assert np.array_equal(nets[1].links, np.array([[0, 0, 1, 1, 0, 1],
                                                   [0, 0, 0, 1, 1, 0],
                                                   [0, 0, 0, 1, 1, 1],
                                                   [0, 0, 1, 0, 1, 1],
                                                   [0, 0, 1, 1, 0, 1],
                                                   [0, 0, 0, 0, 0, 0]]))
    assert np.all(np.isclose(nets[1].weights, np.array([[0, 0, 0.02348412, -0.47971685,  0, 0.0128531],
                                                        [0, 0, 0, -0.78054184, 0.46013031, 0],
                                                        [0, 0, 0, -0.91486681,-0.11334042, 0.78668504],
                                                        [0, 0, 0.32327473, 0, 0.39699364, 0.2487557],
                                                        [0, 0, 0.58691297, 0.96277274, 0, -0.68333513],
                                                        [0, 0, 0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(nets[1].biases, np.array([[0, 0, -5.97746074, 5.44356359, 0.04376756, 5.26342884]]), atol=1e-5))
    assert len(nets[1].actFuns) == 6
    assert nets[1].actFuns[0] is None
    assert nets[1].actFuns[1] is None
    assert nets[1].actFuns[2].to_string() == GaussAct().to_string()
    assert nets[1].actFuns[3].to_string() == Sigmoid().to_string()
    assert nets[1].actFuns[4].to_string() == GaussAct().to_string()
    assert nets[1].aggrFun.to_string() == Sigmoid().to_string()
    assert nets[1].hidden_comp_order is None
    assert nets[1].maxit == 4

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

print()

hd2 = random.randint(0, 3)
print(f"hd1: {hd2}")
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

test_cn_generation()