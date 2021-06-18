import pytest
from ann_point.Functions import *

#TODO sprawdzaj w testach licznosci krotek
#TODO Dopisz wypisywanie eff
#TODO Dopisz F1 score gdzieś

allActFun = [ReLu(), Sigmoid(), TanH(), Softmax(), LReLu(), GaussAct(), SincAct()]

def test_relu():
    # rl = ReLu()
    #
    # arg = np.array([[9], [1], [0], [-1]])
    # exp = np.array([[9], [1], [0], [0]])
    #
    # assert np.array_equal(exp, rl.compute(arg))
    #
    # arg = np.array([[9], [0], [0.25], [-1]])
    # exp = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    #
    # assert np.array_equal(exp, rl.computeDer(arg))
    #
    # assert rl.to_string() == "RL"
    #
    # assert isinstance(rl.copy(), ReLu)

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

    # arg = np.array([[9], [0], [0.25], [-1]])
    # exp = np.array([[1, 0, 0, 0], [0, 0.01, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0.01]])
    #
    # assert np.array_equal(exp, lrl.computeDer(arg))

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
    # assert np.all(np.isclose(gauss.computeDer(arg), np.array([[-0.735758882, 0, 0], [0, 0.073262556, 0], [0, 0, -0.778800783]]), atol=1e-6))
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
    # assert np.all(np.isclose(sinc.computeDer(arg), np.array([[0.116110749 , 0, 0, 0, 0], [0, 0.435397775, 0, 0, 0], [0, 0, 0, 0, 0],
    #                                                          [0, 0, 0, -0.162537031, 0], [0, 0, 0, 0, -0.301168679]]), atol=1e-6))
    assert sinc.to_string() == "SC"
    assert isinstance(sinc.copy(), SincAct)

#TODO test counting generation?
def test_sigma():
    arg = np.array([[3, 2, 1],
                    [0.5, 0, -1],
                    [-3, -10, -100]])
    exp = np.array([[0.952574127, 0.880797078, 0.731058579],
                    [0.622459331, 0.5, 0.268941421],
                    [0.047425873, 4.53979E-05, 3.72008E-44]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.compute(arg), exp, atol=1e-6))
    # assert np.all(np.isclose(sigma.computeDer(arg), np.array([[0.25, 0, 0], [0, 0.196611933, 0], [0, 0, 0.104993585]]), atol=1e-6))
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
    # assert np.all(np.isclose(sm.computeDer(arg), np.array([[0.177759407, -0.145331554, -0.032427853],
    #                                         [-0.145331554, 0.233479597, -0.088148043],
    #                                         [-0.032427853, -0.088148043, 0.120575896]]), atol=1e-5))
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
    # assert np.all(np.isclose(tanh.computeDer(arg), np.array([[0.419974342, 0, 0],
    #                                         [0, 0.070650825, 0],
    #                                         [0, 0, 0.786447733]]), atol=1e-5))
    assert tanh.to_string() == "TH"
    assert isinstance(tanh, TanH)

def test_non_rep_string():
    for i in range(len(allActFun)):
        for j in range(i + 1, len(allActFun)):
            af1 = allActFun[i]
            af2 = allActFun[j]
            assert af1.to_string() != af2.to_string()



