import numpy as np
from ann_point.Functions import *
from ann_point.AnnPoint2 import *
from evolving_classifier.operators.CrossoverOperator import *


def test_crossover():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
    acts = [ReLu(), ReLu(), Sigmoid(), Sigmoid()]
    hlc = [2, 1, 2]
    pointA = AnnPoint2(2, 1, hlc, acts, wei, bias)

    wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
    bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
    acts = [TanH(), Softmax()]
    hlc = [3]
    pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)

    random.seed(10011010)
    np.random.seed(10011010)
    co = SomeCrossoverOperator()
    pointC, pointD = co.crossover(pointA, pointB)

    assert pointC.input_size == 2
    assert pointC.output_size == 1

    assert len(pointC.hidden_neuron_counts) == 3
    assert pointC.hidden_neuron_counts[0] == 2
    assert pointC.hidden_neuron_counts[1] == 1
    assert pointC.hidden_neuron_counts[2] == 2

    assert len(pointC.weights) == 4
    assert np.all(np.isclose(pointC.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
    assert np.all(np.isclose(pointC.weights[1], np.array([[5, 6.0]]), atol=1e-3))
    assert np.all(np.isclose(pointC.weights[2], np.array([[7], [8.0]]), atol=1e-3))
    assert np.all(np.isclose(pointC.weights[3], np.array([[0.22112807, -2.27850884]]), atol=1e-3))

    assert len(pointC.biases) == 4
    assert np.all(np.isclose(pointC.biases[0], np.array([[-1], [-2]]), atol=1e-3))
    assert np.all(np.isclose(pointC.biases[1], np.array([[-3]]), atol=1e-3))
    assert np.all(np.isclose(pointC.biases[2], np.array([[-4], [-5]]), atol=1e-3))
    assert np.all(np.isclose(pointC.biases[3], np.array([[0]]), atol=1e-3))

    assert len(pointC.activation_functions) == 4
    assert pointC.activation_functions[0].to_string() == ReLu().to_string()
    assert pointC.activation_functions[1].to_string() == ReLu().to_string()
    assert pointC.activation_functions[2].to_string() == Sigmoid().to_string()
    assert pointC.activation_functions[3].to_string() == Softmax().to_string()



    wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
    bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
    acts = [TanH(), Softmax()]
    hlc = [3]
    pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)

    assert pointD.input_size == 2
    assert pointD.output_size == 1

    assert len(pointD.hidden_neuron_counts) == 1
    assert pointD.hidden_neuron_counts[0] == 3

    assert len(pointD.weights) == 2
    assert np.all(np.isclose(pointD.weights[0], np.array([[10, 20], [30, 40.0], [50, 60]]), atol=1e-3))
    assert np.all(np.isclose(pointD.weights[1], np.array([[1.08208929, 0.25453426, 1.02913219]]), atol=1e-3))

    assert len(pointD.biases) == 2
    assert np.all(np.isclose(pointD.biases[0], np.array([[-10], [-20.], [-30]]), atol=1e-3))
    assert np.all(np.isclose(pointD.biases[1], np.array([[0]]), atol=1e-3))

    assert len(pointD.activation_functions) == 2
    assert pointD.activation_functions[0].to_string() == TanH().to_string()
    assert pointD.activation_functions[1].to_string() == Sigmoid().to_string()

random.seed(10011010)
np.random.seed(10011010)
print(random.randint(0, 3))
print(get_Xu_matrix((1, 2)))
print(get_Xu_matrix((1, 3)))
test_crossover()