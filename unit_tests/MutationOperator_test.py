import random

import numpy as np
import pytest

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import *
from utility.Mut_Utility import *

def test_simple_mutation():
    hrange = HyperparameterRange((-1, 1), (-1, 1), [ReLu(), Sigmoid(), GaussAct(), TanH()])
    mo = SimpleCNMutation(hrange)

    random.seed(1001)
    np.random.seed(1001)

    link1 = np.array([[0, 1, 1, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 4],
                      [0, 0, 3, 0, 5],
                      [0, 0, 0, 0, 6],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    bia1 = np.array([[-1., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), Sigmoid(), Sigmoid()]

    # link2 = np.array([[0, 0, 0, 0, 0],
    #                   [0, 0, 1, 1, 0],
    #                   [0, 0, 0, 1, 1],
    #                   [0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0]])
    # wei2 = np.array([[0, 0, 0, 0, 0],
    #                  [0, 0, 10, 20, 0],
    #                  [0, 0, 0, 30, 40],
    #                  [0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0]])
    # bia2 = np.array([[-10, -20, -30, -40, -50]])
    # actFuns2 = [None, None, ReLu(), ReLu(), ReLu()]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1, aggrFun=TanH())
    # cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1, aggrFun=TanH())

    mutant = mo.mutate(cn1, pm=0.75, radius=1)

    assert np.array_equal(mutant.links, link1)
    assert np.all(np.isclose(mutant.weights, np.array([[0, 1.54176999, 1.47983043, 0, 5.20238865],
                                                       [0, 0, 2.875572, 0, 5.43418561],
                                                       [0, 0, 0, 0, 5.97519725],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(mutant.bias, np.array([[-1, -1.14810728, -3.75488531, -6.28762932, -4.89076131]]), atol=1e-5))
    assert mutant.actFuns[0] is None
    assert mutant.actFuns[1].to_string() == ReLu().to_string()
    assert mutant.actFuns[2].to_string() == ReLu().to_string()
    assert mutant.actFuns[3].to_string() == Sigmoid().to_string()
    assert mutant.actFuns[4].to_string() == Sigmoid().to_string()

    assert mutant.hidden_comp_order is None

    assert mutant.aggrFun.to_string() == TanH().to_string()



# def test_simple_mutation():
#     hrange = HyperparameterRange((0, 2), (1, 10), [ReLu(), Sigmoid(), Softmax()], [CrossEntropy(), QuadDiff()],
#                                                         (-1, 0), (-3, -2), (-5, -4))
#     pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), Sigmoid()], lossFun=QuadDiff(), learningRate=-1,
#                       momCoeff=-2.5, batchSize=-4)
#
#     mo = SimpleMutationOperator(hrange)
#
#     random.seed(1001)
#     pointB = mo.mutate(pointA, 2, 0.5)
#
#     assert len(pointB.neuronCounts) == 3
#     assert pointB.neuronCounts[0] == 2
#     assert pointB.neuronCounts[1] == 2
#     assert pointB.neuronCounts[2] == 5
#
#     assert len(pointB.actFuns) == 2
#     assert pointB.actFuns[0].to_string() == ReLu().to_string()
#     assert pointB.actFuns[1].to_string() == Softmax().to_string()
#
#     assert pointB.lossFun.to_string() == CrossEntropy().to_string()
#
#     assert pointB.learningRate == pytest.approx(-0.90045, abs=1e-3)
#     assert pointB.momCoeff == pytest.approx(-2.96936, abs=1e-3)
#     assert pointB.batchSize == pytest.approx(-4.12658, abs=1e-3)
#
# def test_add_layer():
#     hrange = HyperparameterRange((0, 2), (1, 10), [ReLu(), Sigmoid(), Softmax()], [CrossEntropy(), QuadDiff()],
#                                              (-1, 0), (-3, -2), (-5, -4))
#     pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), Sigmoid()], lossFun=QuadDiff(), learningRate=-1,
#                   momCoeff=-2.5, batchSize=-4)
#
#     random.seed(1001)
#     pointB = add_layers(pointA, 2, hrange)
#
#     assert len(pointB.neuronCounts) == 6
#     assert pointB.neuronCounts[0] == 2
#     assert pointB.neuronCounts[1] == 4
#     assert pointB.neuronCounts[2] == 3
#     assert pointB.neuronCounts[3] == 4
#     assert pointB.neuronCounts[4] == 3
#     assert pointB.neuronCounts[5] == 5
#
#     assert len(pointB.actFuns) == 5
#     assert pointB.actFuns[0].to_string() == ReLu().to_string()
#     assert pointB.actFuns[1].to_string() == ReLu().to_string()
#     assert pointB.actFuns[2].to_string() == Sigmoid().to_string()
#     assert pointB.actFuns[3].to_string() == Sigmoid().to_string()
#     assert pointB.actFuns[4].to_string() == Sigmoid().to_string()
#
#     assert pointB.lossFun.to_string() == QuadDiff().to_string()
#
#     assert pointB.learningRate == -1
#     assert pointB.momCoeff == -2.5
#     assert pointB.batchSize == -4

# def test_simple_mutation_limits():
#     hrange = HyperparameterRange((0, 3), (1, 10), [ReLu(), Sigmoid(), Softmax()], [CrossEntropy(), QuadDiff()],
#                                  (-1, 0), (-3, -2), (-5, -4))
#     pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), Sigmoid()], lossFun=QuadDiff(), learningRate=-1,
#                       momCoeff=-2.5, batchSize=-4)
#
#     random.seed(1001)
#     mo = SimpleMutationOperator(hrange)
#     pop = []
#     for i in range(200):
#         pop.append(mo.mutate(pointA, 1, 10))
#     layers = [len(pop[i].neuronCounts) for i in range(len(pop))]
#     neurons = [pop[i].neuronCounts[j] for i in range(len(pop)) for j in range(1, len(pop[i].neuronCounts) - 1)]
#     actFuns = [pop[i].actFuns[j] for i in range(len(pop)) for j in range(len(pop[i].actFuns))]
#     lossFuns = [pop[i].lossFun for i in range(len(pop))]
#     lrs = [pop[i].learningRate for i in range(len(pop))]
#     me = [pop[i].momCoeff for i in range(len(pop))]
#     bs = [pop[i].batchSize for i in range(len(pop))]
#
#     assert max(layers) == 5
#     assert min(layers) == 3
#     assert max(neurons) == 10
#     assert min(neurons) == 1
#     assert max(lrs) <= 0
#     assert min(lrs) >= -1
#     assert max(me) <= -2
#     assert min(me) >= -3
#     assert max(bs) <= -4
#     assert min(bs) >= -5
#
#     for i in range(len(hrange.actFunSet)):
#         afh = hrange.actFunSet[i]
#         isthere = False
#         for j in range(len(actFuns)):
#             afp = actFuns[j]
#             isthere = isthere or afh.to_string() == afp.to_string()
#             if isthere == True:
#                 break
#         assert isthere
#
#     hrange.lossFunSet = [CrossEntropy()]
#     for i in range(len(hrange.lossFunSet)):
#         lfh = hrange.lossFunSet[i]
#         isthere = False
#         for j in range(len(lossFuns)):
#             lfp = lossFuns[j]
#             isthere = isthere or lfh.to_string() == lfp.to_string()
#             if isthere == True:
#                 break
#         assert isthere

# def test_some_mutation():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
#     acts = [ReLu(), TanH(), Sigmoid(), TanH()]
#     hlc = [2, 1, 2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#     hrange = HyperparameterRange((0, 4), (1, 4), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     smo = SomeStructMutationOperator(hrange)
#
#     point2 = smo.mutate(point, 1, 2)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 2
#     assert point2.hidden_neuron_counts[0] == 3
#     assert point2.hidden_neuron_counts[1] == 3
#
#     assert len(point2.weights) == 3
#     assert np.all(np.isclose(point2.weights[0], np.array([[1.07541936, 0.57143396], [0.48637152, 0.09957202], [-0.87578195, -0.66401617]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[7, -1.04057717, -0.89370377], [8.0, 0.00577659, 0.25198934], [-0.17240446, 0.40679805, -0.62981052]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[2], np.array([[9, 10, 0.5998332]]), atol=1e-3))
#
#     assert len(point2.biases) == 3
#     assert np.all(np.isclose(point2.biases[0], np.array([[0], [0], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-4], [-5], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[2], np.array([[-6]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 3
#     assert point2.activation_functions[0].to_string() == Sigmoid().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()
#     assert point2.activation_functions[2].to_string() == ReLu().to_string()
#
# def test_layer_increase_func():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3]])]
#     acts = [ReLu(), TanH()]
#     hlc = [2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#     hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     point2 = change_amount_of_layers(point, 3, hrange)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 3
#     assert point2.hidden_neuron_counts[0] == 2
#     assert point2.hidden_neuron_counts[1] == 1
#     assert point2.hidden_neuron_counts[2] == 2
#
#     assert len(point2.weights) == 4
#     assert np.all(np.isclose(point2.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[1.07541936, 0.57143396]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[2], np.array([[0.6878332], [0.1408161]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[3], np.array([[-0.87578195, -0.66401617]]), atol=1e-3))
#
#     assert len(point2.biases) == 4
#     assert np.all(np.isclose(point2.biases[0], np.array([[-1], [-2.0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[0], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[2], np.array([[0], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[3], np.array([[0]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 4
#     assert point2.activation_functions[0].to_string() == ReLu().to_string()
#     assert point2.activation_functions[1].to_string() == Sigmoid().to_string()
#     assert point2.activation_functions[2].to_string() == ReLu().to_string()
#     assert point2.activation_functions[3].to_string() == TanH().to_string()
#
# def test_layer_decrease_func():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8]]), np.array([[9, 10]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3]]), np.array([[-4], [-5]]), np.array([[-6]])]
#     acts = [ReLu(), TanH(), Sigmoid(), TanH()]
#     hlc = [2, 1, 2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#     hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     point2 = change_amount_of_layers(point, 1, hrange)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 1
#     assert point2.hidden_neuron_counts[0] == 2
#
#     assert len(point2.weights) == 2
#     assert np.all(np.isclose(point2.weights[0], np.array([[1.07541936, 0.57143396], [0.48637152, 0.09957202]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[9, 10]]), atol=1e-3))
#
#     assert len(point2.biases) == 2
#     assert np.all(np.isclose(point2.biases[0], np.array([[0], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-6]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 2
#     assert point2.activation_functions[0].to_string() == Sigmoid().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()
#
# def test_layer_increase_nc():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3]])]
#     acts = [ReLu(), TanH()]
#     hlc = [2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     point2 = change_neuron_count_in_layer(point, 0, 3)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 1
#     assert point2.hidden_neuron_counts[0] == 3
#
#     assert len(point2.weights) == 2
#     assert np.all(np.isclose(point2.weights[0], np.array([[1, 2], [3, 4.0], [1.07541936, 0.57143396]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[5, 6.0, 0.39712068]]), atol=1e-3))
#
#     assert len(point2.biases) == 2
#     assert np.all(np.isclose(point2.biases[0], np.array([[-1], [-2.0], [0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 2
#     assert point2.activation_functions[0].to_string() == ReLu().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()
#
# def test_layer_decrease_nc():
#     wei = [np.array([[1, 2], [3, 4.0], [5, 6.0]]), np.array([[7, 8.0, 9]])]
#     bias = [np.array([[-1], [-2.0], [-3.0]]), np.array([[-4]])]
#     acts = [ReLu(), TanH()]
#     hlc = [3]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     point2 = change_neuron_count_in_layer(point, 0, 1)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 1
#     assert point2.hidden_neuron_counts[0] == 1
#
#     assert len(point2.weights) == 2
#     assert np.all(np.isclose(point2.weights[0], np.array([[5, 6.0]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[9.0]]), atol=1e-3))
#
#     assert len(point2.biases) == 2
#     assert np.all(np.isclose(point2.biases[0], np.array([[-3]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-4]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 2
#     assert point2.activation_functions[0].to_string() == ReLu().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()
#
# def test_some_wb_mutation():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.0]])]
#     acts = [ReLu(), TanH()]
#     hlc = [2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#     hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     mo = BiasedGaussianWBMutationOperator(hrange)
#     point2 = mo.mutate(point, 0.75, 2)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 1
#     assert point2.hidden_neuron_counts[0] == 2
#
#     assert len(point2.weights) == 2
#     assert np.all(np.isclose(point2.weights[0], np.array([[1.6878332, 2.1408161], [1.76145729, 4.]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[5., 4.45205966]]), atol=1e-3))
#
#     assert len(point2.biases) == 2
#     assert np.all(np.isclose(point2.biases[0], np.array([[-0.98999465], [-2.]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 2
#     assert point2.activation_functions[0].to_string() == ReLu().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()
#
#
# def test_bias_uni_wb_mutation():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.0]])]
#     acts = [ReLu(), TanH()]
#     hlc = [2]
#     point = AnnPoint2(2, 1, hlc, acts, wei, bias)
#     hrange = HyperparameterRange((0, 4), (1, 3), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()])
#
#     random.seed(10011101)
#     np.random.seed(10011101)
#     mo = BiasedUniformWBMutationOperator(hrange)
#     point2 = mo.mutate(point, 0.75, 2)
#
#     assert point2.input_size == 2
#     assert point2.output_size == 1
#
#     assert len(point2.hidden_neuron_counts) == 1
#     assert point2.hidden_neuron_counts[0] == 2
#
#     assert len(point2.weights) == 2
#     assert np.all(np.isclose(point2.weights[0], np.array([[1.35461979, 3.7321831], [2.33946407, 4.]]), atol=1e-3))
#     assert np.all(np.isclose(point2.weights[1], np.array([[5., 5.62999707]]), atol=1e-3))
#
#     assert len(point2.biases) == 2
#     assert np.all(np.isclose(point2.biases[0], np.array([[-1.95898934], [-2.]]), atol=1e-3))
#     assert np.all(np.isclose(point2.biases[1], np.array([[-3]]), atol=1e-3))
#
#     assert len(point2.activation_functions) == 2
#     assert point2.activation_functions[0].to_string() == ReLu().to_string()
#     assert point2.activation_functions[1].to_string() == TanH().to_string()

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



# random.seed(10011101)
# np.random.seed(10011101)
# print(np.random.random((2, 2)))
# shift = np.random.uniform(-2, 2, size=(2, 2))
# shift[1, 1] = 0
# print(np.array([[1, 2], [3, 4.0]]) + shift)
#
# print(np.random.random((1, 2)))
# shift = np.random.uniform(-2, 2, size=(1, 2))
# shift[0, 0] = 0
# print(np.array([[5, 6.0]]) + shift)
#
# print(np.random.random((2, 1)))
# shift = np.random.uniform(-2, 2, size=(2, 1))
# shift[1, 0] = 0
# print(np.array([[-1], [-2]]) + shift)
#
# print(np.random.random((1, 1)))
# shift = np.random.uniform(-2, 2, size=(1, 1))
# shift[0, 0] = 0
# print(np.array([[-3]]) + shift)

# random.seed(1001)
# print(random.random())
# print(random.randint(0, 1))
# print(choose_without_repetition([1, 2], 1))
# random.random()
# print(get_in_radius(4, min_val=1, max_val=10, radius=0.5))
# random.random()
# print(random.randint(0, 1))
# random.random()
# print(random.randint(0, 1))
# random.random()
# print(random.randint(0, 0))
# random.random()
# print(get_in_radius(-1, -1, 0, 0.5))
# random.random()
# print(get_in_radius(-2.5, -3, -2, 0.5))
# random.random()
# print(get_in_radius(-4, -5, -4, 0.5))
#
# test_simple_mutation()

# random.seed(1001)
# print(random.randint(1, 3))
# print(random.randint(1, 10))
# print(random.randint(0, 2))
# print(random.randint(1, 4))
# print(random.randint(1, 10))
# print(random.randint(0, 2))
# test_add_layer()

# random.seed(1001)
# np.random.seed(1001)
# wei1 = np.array([[0, 1, 2, 0, 4],
#                  [0, 0, 3, 0, 5],
#                  [0, 0, 0, 0, 6],
#                  [0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0]])
# bia1 = np.array([[-1, -2, -3, -4, -5]])
# ch1 = np.zeros((5, 5))
# ch1[np.random.random((5, 5)) <= 0.75] = 1
# print(f"change_prob: \n {ch1}")
# print(f"moves: \n {np.random.normal(wei1, 1, (5,5))}")
# ch1b = np.zeros((1, 5))
# ch1b[np.random.random((1, 5)) <= 0.75] = 1
# print(f"change_prob_b: \n {ch1b}")
# print(f"moves_b: \n {np.random.normal(bia1, 1, (1, 5))}")


