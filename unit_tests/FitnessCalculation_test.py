import pytest

from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from neural_network.ChaosNet import *
from utility.TestingUtility import compare_chaos_network, compare_chaos_networks
from utility.Utility import generate_population, HyperparameterRange
from ann_point.Functions import *

def get_io():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    return inputs, output


def points():
    result = generate_population(HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 3), [ReLu(), Sigmoid(), SincAct()], mut_radius=(0, 1),
                                                     wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.001, 0.002),
                                                     r_prob=(0.1, 0.2)), 2, 2, 3)
    return result


def test_fitness_calculator_with_pure_eff():
    random.seed(1002)
    np.random.seed(1002)
    fc = CNFitnessCalculator()
    anns = points()
    i, o = get_io()
    res = fc.compute(None, to_compute=anns, fitnessFunc=CNFF(), trainInputs=i, trainOutputs=o)

    assert len(res) == 2

    compare_chaos_networks(res[0].net, anns[1])
    assert res[0].ff == pytest.approx(0.30555, abs=1e-3)
    assert res[0].acc == pytest.approx(0.5, abs=1e-3)
    assert res[0].prec == pytest.approx(0.166666, abs=1e-3)
    assert res[0].rec == pytest.approx(0.333333, abs=1e-3)
    assert res[0].f1 == pytest.approx(0.22222, abs=1e-3)
    assert res[0].get_eff() == pytest.approx(0.30555, abs=1e-3)

    compare_chaos_networks(res[1].net, anns[0])#TODO - B - fix this?
    assert res[1].ff == pytest.approx(0.1875, abs=1e-3)
    assert res[1].acc == pytest.approx(0.25, abs=1e-3)
    assert res[1].prec == pytest.approx(0.16666, abs=1e-3)
    assert res[1].rec == pytest.approx(0.16666, abs=1e-3)
    assert res[1].f1 == pytest.approx(0.16666, abs=1e-3)
    assert res[1].get_eff() == pytest.approx(0.1875, abs=1e-3)


#
# random.seed(1002)
# np.random.seed(1002)
#
# nets = points()
# i, o = get_io()
#
# cff = CNFF()
#
# res1 = cff.compute(nets[0], i, o, 1111)
# cm1 = res1[1]
# print(f"f: {res1[0]}")
# print(f"acc: {accuracy(cm1)}")
# print(f"prec: {average_precision(cm1)}")
# print(f"rec: {average_recall(cm1)}")
# print(f"eff: {efficiency(cm1)}")
# print(f"f1: {average_f1_score(cm1)}")
#
# print("\n\n")
#
# res2 = cff.compute(nets[1], i, o, 1111)
# cm2 = res2[1]
# print(f"f: {res2[0]}")
# print(f"acc: {accuracy(cm2)}")
# print(f"prec: {average_precision(cm2)}")
# print(f"rec: {average_recall(cm2)}")
# print(f"eff: {efficiency(cm2)}")
# print(f"f1: {average_f1_score(cm2)}")


test_fitness_calculator_with_pure_eff()







