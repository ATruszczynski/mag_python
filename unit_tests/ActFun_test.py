import pytest
from ann_point.Functions import *

allActFun = [ReLu(), Sigmoid(), TanH(), Softmax(), LReLu(), GaussAct(), SincAct(), Poly2(), Poly3(), Identity()]

def test_relu():
    rl = ReLu()

    arg = np.array([[9, 1, 0],
                    [1, 0, -1],
                    [0, 2, -2]])
    exp = np.array([[9, 1, 0],
                    [1, 0, 0],
                    [0, 2, 0]])

    assert np.array_equal(exp, rl.compute(arg))

    assert rl.to_string() == "RL"

    assert isinstance(rl.copy(), ReLu)


def test_leaky_relu():
    lrl = LReLu()
    arg = np.array([[9, 1, 0],
                    [1, 0, -1],
                    [0, 2, -2]])
    exp = np.array([[9, 1, 0],
                    [1, 0, -0.01],
                    [0, 2, -0.02]])

    assert np.array_equal(exp, lrl.compute(arg))

    assert lrl.to_string() == "L0.01"

    cp = lrl.copy()
    assert isinstance(cp, LReLu)
    assert lrl.to_string() == cp.to_string()
    assert lrl.a == cp.a

    lrl.a = 1

    assert lrl.a == 1
    assert cp.a == 0.01

def test_gauss():
    gauss = GaussAct()

    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.00012341, 0.018315639, 0.367879441],
                    [0.778800783, 1, 0.367879441],
                    [0.00012341, 0, 0]])

    assert np.all(np.isclose(gauss.compute(arg), exp, atol=1e-6))
    assert gauss.to_string() == "GS"
    assert isinstance(gauss.copy(), GaussAct)


def test_sinc():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.047040003, 0.454648713, 0.841470985],
                    [0.958851077, 1, 0.841470985],
                    [0.047040003, -0.054402111, -0.005063656]])

    sinc = SincAct()

    assert np.all(np.isclose(sinc.compute(arg), exp, atol=1e-6))
    assert sinc.to_string() == "SC"
    assert isinstance(sinc.copy(), SincAct)

def test_sigma():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.952574127, 0.880797078, 0.731058579],
                    [0.622459331, 0.5, 0.268941421],
                    [0.047425873, 4.53979E-05, 3.72008E-44]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.compute(arg), exp, atol=1e-6))
    assert sigma.to_string() == "SG"
    assert isinstance(sigma.copy(), Sigmoid)


def test_softmax():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.922029709, 0.880792311, 0.880797078],
                    [0.075684807, 0.119202277, 0.119202922],
                    [0.002285483, 5.41177E-06, 1.20541E-44]])

    sm = Softmax()

    assert np.all(np.isclose(sm.compute(arg), exp, atol=1e-5))
    assert sm.to_string() == "SM"
    assert isinstance(sm.copy(), Softmax)


def test_tanh():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.995054754, 0.96402758, 00.761594156],
                    [0.462117157, 0, -0.761594156],
                    [-0.995054754, -0.999999996, -1]])
    tanh = TanH()

    assert np.all(np.isclose(tanh.compute(arg), exp))
    assert tanh.to_string() == "TH"
    assert isinstance(tanh, TanH)

def test_identity():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    tanh = Identity()

    assert np.all(np.isclose(tanh.compute(arg), exp))
    assert tanh.to_string() == "ID"
    assert isinstance(tanh, Identity)

def test_poly2():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[9, 4, 1],
                    [0.25, 0, 1],
                    [9, 100, 10000]])
    tanh = Poly2()

    assert np.all(np.isclose(tanh.compute(arg), exp))
    assert tanh.to_string() == "P2"
    assert isinstance(tanh, Poly2)

def test_poly3():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[27, 8, 1],
                    [0.125, 0, -1],
                    [-27, -1000, -1000000]])
    tanh = Poly3()

    assert np.all(np.isclose(tanh.compute(arg), exp))
    assert tanh.to_string() == "P3"
    assert isinstance(tanh, Poly3)

def test_non_rep_string():
    for i in range(len(allActFun)):
        for j in range(i + 1, len(allActFun)):
            af1 = allActFun[i]
            af2 = allActFun[j]
            assert af1.to_string() != af2.to_string()
