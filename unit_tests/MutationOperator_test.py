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
    assert point2.hidden_neuron_counts[0] == 3
    assert point2.hidden_neuron_counts[1] == 3

    assert len(point2.weights) == 3
    assert np.all(np.isclose(point2.weights[0], np.array([[1.07541936, 0.57143396], [0.48637152, 0.09957202], [-0.87578195, -0.66401617]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[7, -1.04057717, -0.89370377], [8.0, 0.00577659, 0.25198934], [-0.17240446, 0.40679805, -0.62981052]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[2], np.array([[9, 10, 0.5998332]]), atol=1e-3))

    assert len(point2.biases) == 3
    assert np.all(np.isclose(point2.biases[0], np.array([[0], [0], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-4], [-5], [0]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[2], np.array([[-6]]), atol=1e-3))

    assert len(point2.activation_functions) == 3
    assert point2.activation_functions[0].to_string() == Sigmoid().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()
    assert point2.activation_functions[2].to_string() == ReLu().to_string()

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
    assert np.all(np.isclose(point2.weights[1], np.array([[1.07541936, 0.57143396]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[2], np.array([[0.6878332], [0.1408161]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[3], np.array([[-0.87578195, -0.66401617]]), atol=1e-3))

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
    assert np.all(np.isclose(point2.weights[0], np.array([[1.07541936, 0.57143396], [0.48637152, 0.09957202]]), atol=1e-3))
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
    assert np.all(np.isclose(point2.weights[0], np.array([[1, 2], [3, 4.0], [1.07541936, 0.57143396]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[5, 6.0, 0.39712068]]), atol=1e-3))

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
    mo = BiasedGaussianWBMutationOperator(hrange)
    point2 = mo.mutate(point, 0.75, 2)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 2

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[1.6878332, 2.1408161], [1.76145729, 4.]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[5., 4.45205966]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[-0.98999465], [-2.]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()


def test_bias_uni_wb_mutation():
    wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
    bias = [np.array([[-1], [-2.0]]), np.array([[-3.0]])]
    acts = [ReLu(), TanH()]
    hlc = [2]
    point = AnnPoint2(2, 1, hlc, acts, wei, bias)
    hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])

    random.seed(10011101)
    np.random.seed(10011101)
    mo = BiasedUniformWBMutationOperator(hrange)
    point2 = mo.mutate(point, 0.75, 2)

    assert point2.input_size == 2
    assert point2.output_size == 1

    assert len(point2.hidden_neuron_counts) == 1
    assert point2.hidden_neuron_counts[0] == 2

    assert len(point2.weights) == 2
    assert np.all(np.isclose(point2.weights[0], np.array([[1.35461979, 3.7321831], [2.33946407, 4.]]), atol=1e-3))
    assert np.all(np.isclose(point2.weights[1], np.array([[5., 5.62999707]]), atol=1e-3))

    assert len(point2.biases) == 2
    assert np.all(np.isclose(point2.biases[0], np.array([[-1.95898934], [-2.]]), atol=1e-3))
    assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))

    assert len(point2.activation_functions) == 2
    assert point2.activation_functions[0].to_string() == ReLu().to_string()
    assert point2.activation_functions[1].to_string() == TanH().to_string()

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
# print(choose_without_repetition([1, 2, 3], 2)) # TODO popraw ten test
# print(get_Xu_matrix((2, 2)))
#
# test_layer_decrease_func()

# random.seed(10011101)
# np.random.seed(10011101)
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((1, 1), div=3))
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
# print(get_Xu_matrix((2, 2)))
# print(get_Xu_matrix((2, 2), div=3))
# random.random()
# print(try_choose_different(2, [1, 2, 3, 4]))
# print(get_Xu_matrix((1, 3)))
# print(get_Xu_matrix((1, 1), div=3))
# print("acts")
# random.random()
# print(try_choose_different(TanH(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
# random.random()
# print(try_choose_different(Sigmoid(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
# random.random()
# print(try_choose_different(TanH(), [ReLu(), Sigmoid(), TanH(), Softmax()]).to_string())
# test_some_mutation()

# random.seed(10011101)
# np.random.seed(10011101)
# print(np.random.random((2, 2)))
# shift = np.random.normal(0, 1, size=(2, 2))
# shift[1, 1] = 0
# print(np.array([[1, 2], [3, 4.0]]) + shift)
#
# print(np.random.random((1, 2)))
# shift = np.random.normal(0, 1, size=(1, 2))
# shift[0, 0] = 0
# print(np.array([[5, 6.0]]) + shift)
#
# print(np.random.random((2, 1)))
# shift = np.random.normal(0, 1, size=(2, 1))
# shift[1, 0] = 0
# print(np.array([[-1], [-2]]) + shift)
#
# print(np.random.random((1, 1)))
# shift = np.random.normal(0, 1, size=(1, 1))
# shift[0, 0] = 0
# print(np.array([[-3]]) + shift)
# test_some_wb_mutation()



random.seed(10011101)
np.random.seed(10011101)
print(np.random.random((2, 2)))
shift = np.random.uniform(-2, 2, size=(2, 2))
shift[1, 1] = 0
print(np.array([[1, 2], [3, 4.0]]) + shift)

print(np.random.random((1, 2)))
shift = np.random.uniform(-2, 2, size=(1, 2))
shift[0, 0] = 0
print(np.array([[5, 6.0]]) + shift)

print(np.random.random((2, 1)))
shift = np.random.uniform(-2, 2, size=(2, 1))
shift[1, 0] = 0
print(np.array([[-1], [-2]]) + shift)

print(np.random.random((1, 1)))
shift = np.random.uniform(-2, 2, size=(1, 1))
shift[0, 0] = 0
print(np.array([[-3]]) + shift)

