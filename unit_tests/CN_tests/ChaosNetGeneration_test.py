from utility.Utility import *

#TODO is CN constructor tested?

def test_cn_generation():
    random.seed(1001)
    np.random.seed(1001)

    hrnage = HyperparameterRange((-1, 1), (-10, 10), (1, 5), [ReLu(), GaussAct(), Sigmoid()])
    nets = generate_population(hrange=hrnage, count=2, input_size=2, output_size=1, hidden_size=2)


    assert len(nets) == 2
    assert np.array_equal(nets[0].links, np.array([[0, 0, 1, 1, 1],
                                                   [0, 0, 1, 1, 1],
                                                   [0, 0, 0, 0, 1],
                                                   [0, 0, 1, 0, 1],
                                                   [0, 0, 0, 0, 0]]))
    assert np.all(np.isclose(nets[0].weights, np.array([[0, 0, 0.38607948, -0.64905815,  0.66089761],
                                                        [0, 0, 0.63702411, -0.1645071,  -0.87706191],
                                                        [0, 0, 0, 0,  0.75720675],
                                                        [0, 0, 0.13712337, 0, -0.26970662],
                                                        [0, 0, 0, 0,  0]]), atol=1e-5))
    assert np.all(np.isclose(nets[0].bias, np.array([[-4.10215784, -1.57532018, -5.00369496, -1.15930867, 1.6677175]]), atol=1e-5))
    assert nets[0].actFuns[0] is None
    assert nets[0].actFuns[1] is None
    assert nets[0].actFuns[2].to_string() == ReLu().to_string()
    assert nets[0].actFuns[3].to_string() == ReLu().to_string()
    assert nets[0].actFuns[4].to_string() == ReLu().to_string()
    assert nets[0].aggrFun.to_string() == Sigmoid().to_string()
    assert nets[0].hidden_comp_order is None
    assert nets[0].maxit == 4

    assert np.array_equal(nets[1].links, np.array([[0, 0, 1, 1, 1],
                                                   [0, 0, 1, 1, 1],
                                                   [0, 0, 0, 1, 0],
                                                   [0, 0, 1, 0, 0],
                                                   [0, 0, 0, 0, 0]]))
    assert np.all(np.isclose(nets[1].weights, np.array([[0, 0, 0.88576578, 0.58691297, 0.96277274],
                                                        [0, 0, 0.45904655, 0.50014929, 0.66848079],
                                                        [0, 0, 0, 0.83449345, 0],
                                                        [0, 0, 0.00437676, 0, 0],
                                                        [0, 0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(nets[1].bias, np.array([[-8.54177062, -3.11764183, -3.6306767, -2.24112573, 5.66324156]]), atol=1e-5))
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

print(f"den1: {random.random()}")
print(f"lin_prob1: \n{np.random.random((5, 5))}")
print(f"wei_1: \n{np.random.uniform(-1, 1, (5, 5))}")
print(f"bia_1: \n{np.random.uniform(-10, 10, (1, 5))}")
print(f"af1_2: {random.randint(0, 2)}")
print(f"af1_3: {random.randint(0, 2)}")
print(f"af1_4: {random.randint(0, 2)}")
print(f"aggrf_1: {random.randint(0, 2)}")
print(f"maxit_1: {random.randint(1, 5)}")

print()

print(f"den2: {random.random()}")
print(f"lin_prob2: \n{np.random.random((5, 5))}")
print(f"wei_2: \n{np.random.uniform(-1, 1, (5, 5))}")
print(f"bia_2: \n{np.random.uniform(-10, 10, (1, 5))}")
print(f"af2_2: {random.randint(0, 2)}")
print(f"af2_3: {random.randint(0, 2)}")
print(f"af2_4: {random.randint(0, 2)}")
print(f"aggrf_2: {random.randint(0, 2)}")
print(f"maxit_2: {random.randint(1, 5)}")

# test_cn_generation()