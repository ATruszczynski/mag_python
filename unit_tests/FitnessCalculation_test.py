import pytest

from evolving_classifier.LsmFitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from neural_network.LsmNetwork import *
from utility.TestingUtility import assert_chaos_network_properties, assert_chaos_networks_same
from utility.Utility import generate_population, HyperparameterRange
from ann_point.Functions import *

def get_io():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    return inputs, output


def points():
    result = generate_population(HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 3), [ReLu(), Sigmoid(), SincAct()], mut_radius=(-10, 0),
                                                     swap=(-0.05, 0), multi=(-0.6, 0), p_prob=(-0.4, 0.), c_prob=(-0.001, 0.),
                                                     p_rad=(-0.1, 0)), 2, 2, 3)
    return result


def test_fitness_calculator_with_pure_eff():
    random.seed(1002)
    np.random.seed(1002)
    fc = LsmFitnessCalculator()
    anns = points()
    i, o = get_io()
    res = fc.compute(None, to_compute=anns, fitnessFunc=CNFF(), trainInputs=i, trainOutputs=o)

    anns[0].run(np.hstack(i))
    anns[1].run(np.hstack(i))

    assert len(res) == 2

    assert_chaos_networks_same(res[0].net, anns[1])
    assert len(res[0].ff) == 1
    assert res[0].ff[0] == pytest.approx(0.3055555555555555, abs=1e-3)
    assert res[0].get_acc() == pytest.approx(0.5, abs=1e-3)
    assert res[0].get_avg_prec() == pytest.approx(0.1666666, abs=1e-3)
    assert res[0].get_avg_rec() == pytest.approx(0.3333333, abs=1e-3)
    assert res[0].get_avg_f1() == pytest.approx(0.222222222, abs=1e-3)
    assert res[0].get_eff() == pytest.approx(0.30555555, abs=1e-3)
    assert res[0].get_meff() == pytest.approx(0.0, abs=1e-3)

    assert_chaos_networks_same(res[1].net, anns[0])
    assert len(res[1].ff) == 1
    assert res[1].ff[0] == pytest.approx(0.1875, abs=1e-3)
    assert res[1].get_acc() == pytest.approx(0.25, abs=1e-3)
    assert res[1].get_avg_prec() == pytest.approx(0.16666, abs=1e-3)
    assert res[1].get_avg_rec() == pytest.approx(0.16666, abs=1e-3)
    assert res[1].get_avg_f1() == pytest.approx(0.16666, abs=1e-3)
    assert res[1].get_eff() == pytest.approx(0.1875, abs=1e-3)
    assert res[1].get_meff() == pytest.approx(0.0, abs=1e-3)








