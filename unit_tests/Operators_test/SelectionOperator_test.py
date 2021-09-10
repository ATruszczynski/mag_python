import random

import pytest

# from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from evolving_classifier.operators.SelectionOperator import *
from utility.CNDataPoint import CNDataPoint
from utility.MockCN import MockCN
from utility.Utility import compare_lists


def test_roullette_select():
    random.seed(2020)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(3, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = 9.
    adp2 = CNDataPoint(cn2)
    adp2.ff = 6.
    adp3 = CNDataPoint(cn3)
    adp3.ff = 3.
    adp4 = CNDataPoint(cn4)
    adp4.ff = 12.

    list = [adp1, adp2, adp3, adp4]

    so = RoulletteSelection()

    assert adp4.net.to_string() == so.select(list).to_string()
    assert adp1.net.to_string() == so.select(list).to_string()

def test_select():
    random.seed(1010)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(3, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = 1.8
    adp2 = CNDataPoint(cn2)
    adp2.ff = 1.5
    adp3 = CNDataPoint(cn3)
    adp3.ff = 2.4
    adp4 = CNDataPoint(cn4)
    adp4.ff = 5.4

    list = [adp1, adp2, adp3, adp4]

    so = TournamentSelection(2)

    chosen = so.select(list)


    assert adp4.net.to_string() == chosen.to_string()

    adp4.net.mat[0][1] = 3
    adp4.net.neuron_count = -1

    assert adp4.net.to_string() == (MockCN(-1, np.array([[4, 3, 4, 4]]))).to_string()
    assert chosen.to_string() == (MockCN(4, np.array([[4, 4, 4, 4]]))).to_string()

def test_select_05_1():
    random.seed(1010)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(3, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))
    cn5 = MockCN(5, np.array([[5, 5, 5, 5, 5]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = [1.8, -1]
    adp2 = CNDataPoint(cn2)
    adp2.ff = [5.4, -2]
    adp3 = CNDataPoint(cn3)
    adp3.ff = [2.4, -3]
    adp4 = CNDataPoint(cn4)
    adp4.ff = [5.4, -4]
    adp5 = CNDataPoint(cn5)
    adp5.ff = [5.4, -5]

    list = [adp1, adp2, adp3, adp4, adp5]

    so = TournamentSelection95(3)

    chosen = so.select(list)


    assert adp2.net.to_string() == chosen.to_string()

    adp2.net.mat[0][1] = 3
    adp2.net.neuron_count = -1

    assert adp2.net.to_string() == (MockCN(-1, np.array([[2, 3]]))).to_string()
    assert chosen.to_string() == (MockCN(2, np.array([[2, 2]]))).to_string()

def test_select_05_2():
    random.seed(19611)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(6, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))
    cn5 = MockCN(5, np.array([[5, 5, 5, 5, 5]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = [1.8, -1]
    adp2 = CNDataPoint(cn2)
    adp2.ff = [5.4, -2]
    adp3 = CNDataPoint(cn3)
    adp3.ff = [5.0, -6]
    adp4 = CNDataPoint(cn4)
    adp4.ff = [5.0, -4]
    adp5 = CNDataPoint(cn5)
    adp5.ff = [5.4, -5]

    list = [adp1, adp2, adp3, adp4, adp5]

    so = TournamentSelection95(3)

    chosen = so.select(list)

    assert adp3.net.to_string() == chosen.to_string()

def test_select_05_3():
    random.seed(5888)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(6, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))
    cn5 = MockCN(5, np.array([[5, 5, 5, 5, 5]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = [1.8, -1]
    adp2 = CNDataPoint(cn2)
    adp2.ff = [5.4, -2]
    adp3 = CNDataPoint(cn3)
    adp3.ff = [5.0, -6]
    adp4 = CNDataPoint(cn4)
    adp4.ff = [5.0, -4]
    adp5 = CNDataPoint(cn5)
    adp5.ff = [5.4, -5]

    list = [adp1, adp2, adp3, adp4, adp5]

    so = TournamentSelection95(3)

    chosen = so.select(list)

    assert adp3.net.to_string() == chosen.to_string()

def test_select_05_4():
    random.seed(5888)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(6, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))
    cn5 = MockCN(5, np.array([[5, 5, 5, 5, 5]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = [0, 1.8, -1]
    adp2 = CNDataPoint(cn2)
    adp2.ff = [0, 5.4, -2]
    adp3 = CNDataPoint(cn3)
    adp3.ff = [0, 5.0, -6]
    adp4 = CNDataPoint(cn4)
    adp4.ff = [0, 5.0, -4]
    adp5 = CNDataPoint(cn5)
    adp5.ff = [0, 5.4, -5]

    list = [adp1, adp2, adp3, adp4, adp5]

    so = TournamentSelection95(3)

    chosen = so.select(list)

    assert adp3.net.to_string() == chosen.to_string()


def test_select_too_few():
    random.seed(1010)

    cn1 = MockCN(1, np.array([[1]]))
    cn2 = MockCN(2, np.array([[2, 2]]))
    cn3 = MockCN(3, np.array([[3, 3, 3]]))
    cn4 = MockCN(4, np.array([[4, 4, 4, 4]]))

    adp1 = CNDataPoint(cn1)
    adp1.ff = 1.8
    adp2 = CNDataPoint(cn2)
    adp2.ff = 7.5
    adp3 = CNDataPoint(cn3)
    adp3.ff = 2.4
    adp4 = CNDataPoint(cn4)
    adp4.ff = 5.4

    list = [adp1, adp2, adp3, adp4]

    so = TournamentSelection(10)

    assert adp2.net.to_string() == so.select(list).to_string()
