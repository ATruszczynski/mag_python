import random

from ann_point.AnnPoint import AnnPoint
from evolving_classifier.EvolvingClassifier import *


def test_roullette_select():
    random.seed(2020)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    list = [[ap1, 3.],
            [ap2, 2.],
            [ap3, 1.],
            [ap4, 4.]]

    so = RoulletteSelection()

    assert ap4.to_string() == so.select(list).to_string()
    assert ap1.to_string() == so.select(list).to_string()

def test_select():
    random.seed(1010)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    list = [[ap1, 0.6],
            [ap2, 0.5],
            [ap3, 0.8],
            [ap4, 1.8]]

    so = TournamentSelection(2)

    assert ap4.to_string() == so.select(list).to_string()



def test_select_too_few():
    random.seed(1010)

    ap1 = AnnPoint([1, 1], [ReLu()], QuadDiff(), 1, 1, 1)
    ap2 = AnnPoint([2, 2, 2], [Sigmoid(), Sigmoid()], QuadDiff(), 2, 2, 2)
    ap3 = AnnPoint([3, 3, 3, 3], [TanH(), TanH(), TanH()], CrossEntropy(), 3, 3, 3)
    ap4 = AnnPoint([4, 4, 4, 4, 4], [Softmax(), Softmax(), Softmax(), Softmax()], CrossEntropy(), 4, 4, 4)

    list = [[ap1, 0.6],
            [ap2, 2.5],
            [ap3, 0.8],
            [ap4, 1.8]]

    so = TournamentSelection(10)

    assert ap2.to_string() == so.select(list).to_string()