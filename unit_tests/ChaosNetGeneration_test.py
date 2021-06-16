from utility.Utility import *


def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), [ReLu(), GaussAct(), Sigmoid()])
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1, hidden_size=2)


    assert len(nets) == 2
    assert np.array_equal(nets[0].links, np.array([[0, 0, 1, 1, 1],
                                                   [0, 0, 1, 1, 1],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 0, 0, 0, 0]]))
    assert np.all(np.isclose(nets[0].weights, np.array([[0,  0,  0.38607948, -0.64905815,  0.66089761],
                                                        [0,  0,  0.63702411, -0.1645071,  -0.87706191],
                                                        [0,  0, 0, 0,  0.75720675],
                                                        [0, 0,  0, 0, -0.26970662],
                                                        [0, 0,  0, 0,  0]]), atol=1e-5))
    assert np.all(np.isclose(nets[0].bias, np.array([[-4.10215784, -1.57532018, -5.00369496, -1.15930867, 1.6677175]]), atol=1e-5))
    assert nets[0].actFuns[0] is None
    assert nets[0].actFuns[1] is None
    assert nets[0].actFuns[2].to_string() == ReLu().to_string()
    assert nets[0].actFuns[3].to_string() == ReLu().to_string()
    assert nets[0].actFuns[4].to_string() == ReLu().to_string()

# random.seed(1001)
# np.random.seed(1001)
#
# print(f"den1: {random.random()}")
# print(f"lin_prob1: \n{np.random.random((5, 5))}")
# print(f"wei_1: \n{np.random.uniform(-1, 1, (5, 5))}")
# print(f"bia_1: \n{np.random.uniform(-10, 10, (1, 5))}")
# print(f"af1_2: {random.randint(0, 2)}")
# print(f"af1_3: {random.randint(0, 2)}")
# print(f"af1_4: {random.randint(0, 2)}")
#
# test_cn_generation()