import random

import pytest

from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from evolving_classifier.operators.SelectionOperator import *
from utility.AnnDataPoint import AnnDataPoint


def test_roullette_select():
    random.seed(2020)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    adp1 = AnnDataPoint(ap1)
    adp1.ff = 9.
    adp2 = AnnDataPoint(ap2)
    adp2.ff = 6.
    adp3 = AnnDataPoint(ap3)
    adp3.ff = 3.
    adp4 = AnnDataPoint(ap4)
    adp4.ff = 12.

    list = [[ap1, adp1],
            [ap2, adp2],
            [ap3, adp3],
            [ap4, adp4]]

    so = RoulletteSelection()

    assert ap4.to_string() == so.select(list).to_string()
    assert ap1.to_string() == so.select(list).to_string()

def test_select():
    random.seed(1010)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    adp1 = AnnDataPoint(ap1)
    adp1.ff = 1.8
    adp2 = AnnDataPoint(ap2)
    adp2.ff = 1.5
    adp3 = AnnDataPoint(ap3)
    adp3.ff = 2.4
    adp4 = AnnDataPoint(ap4)
    adp4.ff = 5.4

    list = [[ap1, adp1],
            [ap2, adp2],
            [ap3, adp3],
            [ap4, adp4]]

    so = TournamentSelection(2)

    assert ap4.to_string() == so.select(list).to_string()



def test_select_too_few():
    random.seed(1010)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    adp1 = AnnDataPoint(ap1)
    adp1.ff = 1.8
    adp2 = AnnDataPoint(ap2)
    adp2.ff = 7.5
    adp3 = AnnDataPoint(ap3)
    adp3.ff = 2.4
    adp4 = AnnDataPoint(ap4)
    adp4.ff = 5.4

    list = [[ap1, adp1],
            [ap2, adp2],
            [ap3, adp3],
            [ap4, adp4]]

    so = TournamentSelection(10)

    assert ap2.to_string() == so.select(list).to_string()

random.seed(1010)
print(choose_without_repetition([0, 1, 2, 3], 2))

test_select()