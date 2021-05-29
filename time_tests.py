from ann_point.Functions import *

from ann_point.AnnPoint2 import AnnPoint2
from evolving_classifier.operators.MutationOperators import *
import numpy as np
import time


def compare_mutations():
    smo = SomeWBMutationOperator(get_default_hrange())
    smo2 = SomeWBMutationOperator2(get_default_hrange())

    point = AnnPoint2(10000, 1000, weights=[np.zeros((1000, 10000))], biases=[np.zeros((1000, 1))], activationFuns=[ReLu()], hiddenNeuronCounts=[])

    n = 100

    s = time.time()
    for i in range(n):
        _ = smo2.mutate(point, 1, 2)
    t = time.time()

    print(round((t - s)/n, 2))


    s = time.time()
    for i in range(n):
        _ = smo.mutate(point, 1, 2)
    t = time.time()

    print(round((t - s)/n, 2))

compare_mutations()