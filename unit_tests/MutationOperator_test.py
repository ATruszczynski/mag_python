import random

import numpy as np

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import *
from utility.Mut_Utility import *


def test_some_mutation():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
    acts = [ReLu(), TanH(), Sigmoid(), TanH()]
    hlc = [2, 1, 2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)
    hrange = HyperparameterRange((0, 4), (1, 4), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])

    random.seed(10011101)
    np.random.seed(10011101)
    smo = SomeStructMutationOperator(hrange)

    point2 = smo.mutate(point, 1, 2)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 2
    assert point2.hidden_neuron_counts[0] == 2
    assert point2.hidden_neuron_counts[1] == 3

    assert len(point2.weights) == 3
    assert np.all(np.isclose(point2.weights[0], np.array([[0.08805, 0.6352], [0.58163321, 0.49955724]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[7, -2.2785], [8.0, 1.3920], [0.58856006, 0.59203895]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[2], np.array([[9, 10, -0.7113]]), atol=1e-3))

    assert len(point2.biases) == 3
    assert np.all(np.isclose(point2.biases[0], np.array([[0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-4], [-5], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[2], np.array([[-6]]), atol=1e-3))

    assert len(point2.activation_functions) == 3
    assert point2.activation_functions[0].to_string() == Softmax().to_string()
    assert point2.activation_functions[1].to_string() == ReLu().to_string()
    assert point2.activation_functions[2].to_string() == Softmax().to_string()

def test_layer_increase_func():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3]])]
    acts = [ReLu(), TanH()]
    hlc = [2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)
    hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])

    random.seed(10011101)
    np.random.seed(10011101)
    point2 = change_amount_of_layers(point, 3, hrange)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 3
    assert point2.hidden_neuron_counts[0] == 2
    assert point2.hidden_neuron_counts[1] == 1
    assert point2.hidden_neuron_counts[2] == 2

    assert len(point2.weights) == 4
    assert np.all(np.isclose(point2.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[-0.5462, -0.2360]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[2], np.array([[0.8225], [0.7064]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[3], np.array([[-1.6111, 0.9843]]), atol=1e-3))

    assert len(point2.biases) == 4
    assert np.all(np.isclose(point2.biases[0], np.array([[-1], [-2.0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[2], np.array([[0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[3], np.array([[0]]), atol=1e-3))

    assert len(point2.activation_functions) == 4
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == Sigmoid().to_string()
    assert point2.activation_functions[2].to_string() == ReLu().to_string()
    assert point2.activation_functions[3].to_string() == TanH().to_string()

def test_layer_decrease_func():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8]]), np.array([[9, 10]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3]]), np.array([[-4], [-5]]), np.array([[-6]])]
    acts = [ReLu(), TanH(), Sigmoid(), TanH()]
    hlc = [2, 1, 2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)
    hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])

    random.seed(10011101)
    np.random.seed(10011101)
    point2 = change_amount_of_layers(point, 1, hrange)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 2

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[0.2188, 0.7651], [-0.5462, -0.2360]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[9, 10]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-6]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == Sigmoid().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()

def test_layer_increase_nc():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3]])]
    acts = [ReLu(), TanH()]
    hlc = [2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)

    random.seed(10011101)
    np.random.seed(10011101)
    point2 = change_neuron_count_in_layer(point, 0, 3)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 3

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[1, 2], [3, 4.0], [-0.1158, 0.1440]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[5, 6.0, 0.1245]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[-1], [-2.0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()

def test_layer_decrease_nc():
    wei = [np.array([[1, 2], [3, 4.0], [5, 6.0]]), np.array([[7, 8.0, 9]])]
    bias = [np.array([[-1], [-2.0], [-3.0]]), np.array([[-4]])]
    acts = [ReLu(), TanH()]
    hlc = [3]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)

    random.seed(10011101)
    np.random.seed(10011101)
    point2 = change_neuron_count_in_layer(point, 0, 1)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 1

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[5, 6.0]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[9.0]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[-3]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-4]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()

def test_some_wb_mutation():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3.0]])]
    acts = [ReLu(), TanH()]
    hlc = [2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)
    hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])

    random.seed(10011101)
    np.random.seed(10011101)
    mo = SomeWBMutationOperator(hrange)
    point2 = mo.mutate(point, 1, 2)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 2

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[2.40716, 2.3014], [8.1330, 6.1808]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[5.1038, 4.5587]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[-2.4226], [-0.2861]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-5.6206]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()





#
# random.seed(10011101)
# np.random.seed(10011101)
# print(random.randint(1, 3))
# print(random.randint(0, 3))
# print(random.randint(1, 3))
# print(random.randint(0, 3))
# print(random.randint(0, 1))
# print(random.randint(0, 2))
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((2, 1)))
# print(get_Xu_matrix((1, 2)))
# test_layer_increase_func()

# random.seed(10011101)
# np.random.seed(10011101)
# print(choose_without_repetition([0, 1, 2], 2)) # TODO popraw ten test
# print(get_Xu_matrix((2, 2)))
#
# test_layer_decrease_func()

# random.seed(10011101)
# np.random.seed(10011101)
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((1, 1)))
#
# test_layer_increase_nc()

# random.seed(10011101)
# np.random.seed(10011101)
# print(choose_without_repetition([0, 1, 2], 2))
#
# test_layer_decrease_nc()
#
# random.seed(10011101)
# np.random.seed(10011101)
# random.random()
# print(random.randint(0, 1))
# print(choose_without_repetition([1, 2, 3], 1))
# print(get_Xu_matrix((1, 2)))
# print("neuron counts")
# random.random()
# print(try_choose_different(1, [1, 2, 3, 4]))
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((2, 1)))
# random.random()
# print(try_choose_different(2, [1, 2, 3, 4]))
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((1, 1)))
# print("acts")
# random.random()
# print(try_choose_different(TanH(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
# random.random()
# print(try_choose_different(Sigmoid(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
# random.random()
# print(try_choose_different(TanH(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
#
# test_some_mutation()

random.seed(10011101)
np.random.seed(10011101)
random.random()
print(1 + random.gauss(0, 2))
random.random()
print(2 + random.gauss(0, 2))
random.random()
print(3 + random.gauss(0, 2))
random.random()
print(4 + random.gauss(0, 2))
random.random()
print(5 + random.gauss(0, 2))
random.random()
print(6 + random.gauss(0, 2))
random.random()
print(-1 + random.gauss(0, 2))
random.random()
print(-2 + random.gauss(0, 2))
random.random()
print(-3 + random.gauss(0, 2))
test_some_wb_mutation()

