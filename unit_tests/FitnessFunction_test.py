import random
import numpy as np
import pytest

from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from evolving_classifier.FitnessFunction import PureEfficiencyFF
from unit_tests.ANN_test_test import get_io, network_from_point, efficiency

def get_point():
    return AnnPoint(neuronCounts=[2, 3, 3], actFuns=[ReLu(), Sigmoid()], lossFun=QuadDiff(), learningRate=-1, momCoeff=-2, batchSize=-3)


def test_pure_fitness_function():
    point = get_point()
    i, o = get_io()
    ff = PureEfficiencyFF(2)
    res = ff.compute(point, i, o, 1001)

    assert res[0] == pytest.approx(0.3333, abs=1e-3)
    assert res[1] == pytest.approx(0.3333, abs=1e-3)

seed = 1001
random.seed(seed)
np.random.seed(seed)
point = get_point()
i, o = get_io()

network = network_from_point(point, seed)
network.train(i, o, 2)
test_res = network.test(i, o)
print(test_res[:3])
print(test_res[3])
print(efficiency(test_res[3]))

test_pure_fitness_function()