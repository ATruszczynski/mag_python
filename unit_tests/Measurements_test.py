from neural_network.FeedForwardNeuralNetwork import *

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


