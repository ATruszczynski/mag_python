import pytest

from neural_network.ChaosNet import *
# from neural_network.FeedForwardNeuralNetwork import *

def test_accuracy():
    conf = np.array([[1, 0, 1, 0], [0, 2, 0, 0], [1, 2, 2, 0], [0, 0, 0, 1]])

    acc = accuracy(confusion_matrix=conf)

    assert acc == 0.6

def test_average_precision():
    conf = np.array([[1, 0, 0, 0], [0, 2, 0, 3], [0, 2, 0, 0], [1, 0, 0, 1]])
    prec = average_precision(conf)

    assert prec == 5/16

    conf = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 3], [1, 0, 0, 1]])
    prec = average_precision(conf)

    assert prec == 1 / 6

def test_average_recall():
    conf = np.array([[1, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 2, 0, 4]])
    prec = average_recall(conf)

    assert prec == 6/9

    conf = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 2, 0, 4]])
    prec = average_recall(conf)

    assert prec == 13 / 24

def test_average_f1_scores():
    conf = np.array([[1, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 2, 0, 4]])
    prec = average_f1_score(conf)

    assert prec == pytest.approx(0.71111, abs=1e-4)

    conf = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 2, 0, 4]])
    prec = average_f1_score(conf)

    assert prec == pytest.approx(0.53333, abs=1e-4)

def test_mefficiency():
    conf = np.array([[1, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 2, 0, 4]])
    prec = m_efficiency(conf)

    assert prec == pytest.approx(0.333333333, abs=1e-4)

    conf = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 2, 0, 4]])
    prec = m_efficiency(conf)

    assert prec == pytest.approx(0.0, abs=1e-4)


# conf = np.array([[1, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 2, 0, 4]])
# conf2 = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 2, 0, 4]])
#
# # print( min([accuracy(conf), get_precisions(conf), get_recalls(conf), get_f1_scores(conf)]) )
# # print( min([accuracy(conf2), get_precisions(conf2), get_recalls(conf2), get_f1_scores(conf2)]) )
#
# c = conf2
# all = [accuracy(c)]
# all.extend(get_precisions(c))
# all.extend(get_recalls(c))
# all.extend(get_f1_scores(c))
#
# print(min(all))
#
# test_mefficiency()

