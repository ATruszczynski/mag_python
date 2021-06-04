import multiprocessing as mp
import random

import numpy as np
import pytest

from ann_point.AnnPoint import *
from evolving_classifier.FitnessCalculator import OnlyFitnessCalculator
from evolving_classifier.FitnessFunction import *
from neural_network.FeedForwardNeuralNetwork import network_from_point
from unit_tests.ANN_test_test import get_io


def points():
    result = []
    result.append(AnnPoint(neuronCounts=[2, 3, 3], actFuns=[ReLu(), Sigmoid()], lossFun=QuadDiff(), learningRate=-1, momCoeff=-2, batchSize=-3))
    result.append(AnnPoint(neuronCounts=[2, 4, 3], actFuns=[TanH(), ReLu()], lossFun=QuadDiff(), learningRate=-1, momCoeff=-2, batchSize=-3))

    return result


def test_fitness_calculator_with_pure_eff():
    fc = OnlyFitnessCalculator([1, 0.5])
    anns = points()
    i, o = get_io()
    random.seed(1002)
    np.random.seed(1002)
    res = fc.compute(None,to_compute=anns, fitnessFunc=PureEfficiencyFF(2), trainInputs=i, trainOutputs=o)

    assert len(res) == 2

    assert len(res[0]) == 2
    assert res[0][0].to_string() == anns[1].to_string()
    assert res[0][1].ff == pytest.approx(0.31018, abs=1e-3)
    assert res[0][1].acc == pytest.approx((0.5 + 0.25)/2, abs=1e-3)
    assert res[0][1].prec == pytest.approx((1/3 + 1/9)/2, abs=1e-3)
    assert res[0][1].rec == pytest.approx((1/2 + 1/6)/2, abs=1e-3)
    assert res[0][1].touch == pytest.approx(2, abs=1e-3)
    assert res[0][1].get_eff() == pytest.approx(0.31018, abs=1e-3)

    assert len(res[1]) == 2
    assert res[1][0].to_string() == anns[0].to_string()
    assert res[1][1].ff == pytest.approx(0.33333, abs=1e-3)
    assert res[1][1].acc == pytest.approx(1/2, abs=1e-3)
    assert res[1][1].prec == pytest.approx(1/6, abs=1e-3)
    assert res[1][1].rec == pytest.approx(1/3, abs=1e-3)
    assert res[1][1].touch == pytest.approx(1, abs=1e-3)
    assert res[1][1].get_eff() == pytest.approx(0.33333, abs=1e-3)


def test_fitness_calculator_with_prog_eff():
    fc = OnlyFitnessCalculator([1, 0.5])
    anns = points()
    i, o = get_io()
    random.seed(1002)
    np.random.seed(1002)
    res = fc.compute(None,to_compute=anns, fitnessFunc=ProgressFF(2), trainInputs=i, trainOutputs=o)

    assert len(res) == 2
    assert len(res[0]) == 2
    assert len(res[1]) == 2

    assert res[0][0].to_string() == anns[1].to_string()
    assert res[0][1].ff == pytest.approx(0.11631, abs=1e-3)
    assert res[0][1].acc == pytest.approx((0.5 + 0.25)/2, abs=1e-3)
    assert res[0][1].prec == pytest.approx((1/3 + 1/9)/2, abs=1e-3)
    assert res[0][1].rec == pytest.approx((1/2 + 1/6)/2, abs=1e-3)
    assert res[0][1].touch == pytest.approx(2, abs=1e-3)
    assert res[0][1].get_eff() == pytest.approx(0.31018, abs=1e-3)

    assert res[1][0].to_string() == anns[0].to_string()
    assert res[1][1].ff == pytest.approx(0.125, abs=1e-3)
    assert res[1][1].acc == pytest.approx(1/2, abs=1e-3)
    assert res[1][1].prec == pytest.approx(1/6, abs=1e-3)
    assert res[1][1].rec == pytest.approx(1/3, abs=1e-3)
    assert res[1][1].touch == pytest.approx(1, abs=1e-3)
    assert res[1][1].get_eff() == pytest.approx(0.33333, abs=1e-3)


seed = 1002
random.seed(seed)
np.random.seed(seed)

i, o = get_io()

annpoints = points()
seed1 = random.randint(0, 1000)
seed2 = random.randint(0, 1000)
print(seed1)
print(seed2)

fit = ProgressFF(2)

fit_1_1 = fit.compute(annpoints[0], i, o, seed1)
fit_1_2 = fit.compute(annpoints[1], i, o, seed2)
print(fit_1_1[0])
print(fit_1_1[1])
print(fit_1_2[0])
print(fit_1_2[1])

seed3 = random.randint(0, 1000)
print(seed3)
fit_2_2 = fit.compute(annpoints[1], i, o, seed3)
print(fit_2_2[0])
print(fit_2_2[1])

print()
print(fit_1_1[0])
print((fit_1_2[0] + fit_2_2[0]) / 2)

test_fitness_calculator_with_pure_eff()