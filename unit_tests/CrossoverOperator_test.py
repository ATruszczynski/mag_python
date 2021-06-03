import numpy as np
from ann_point.Functions import *
from ann_point.AnnPoint2 import *
from evolving_classifier.operators.CrossoverOperator import *
from utility.Mut_Utility import resize_layer

def test_simple_crossover():
    pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), TanH()], lossFun=QuadDiff(), learningRate=-1,
                      momCoeff=-2, batchSize=-3)
    pointB = AnnPoint(neuronCounts=[2, 30, 5], actFuns=[TanH(), Softmax()], lossFun=CrossEntropy(), learningRate=-10,
                      momCoeff=-20, batchSize=-30)

    random.seed(1001)
    co = SimpleCrossoverOperator()

    pointC, pointD = co.crossover(pointA, pointB)

    assert len(pointC.neuronCounts) == 4
    assert pointC.neuronCounts[0] == 2
    assert pointC.neuronCounts[1] == 3
    assert pointC.neuronCounts[2] == 4
    assert pointC.neuronCounts[3] == 5

    assert len(pointC.actFuns) == 3
    assert pointC.actFuns[0].to_string() == ReLu().to_string()
    assert pointC.actFuns[1].to_string() == Sigmoid().to_string()
    assert pointC.actFuns[2].to_string() == TanH().to_string()

    assert pointC.lossFun.to_string() == CrossEntropy().to_string()

    assert pointC.learningRate == -1
    assert pointC.momCoeff == -20
    assert pointC.batchSize == -30



    assert len(pointD.neuronCounts) == 3
    assert pointD.neuronCounts[0] == 2
    assert pointD.neuronCounts[1] == 30
    assert pointD.neuronCounts[2] == 5

    assert len(pointD.actFuns) == 2
    assert pointD.actFuns[0].to_string() == TanH().to_string()
    assert pointD.actFuns[1].to_string() == Softmax().to_string()

    assert pointD.lossFun.to_string() == QuadDiff().to_string()

    assert pointD.learningRate == -10
    assert pointD.momCoeff == -2
    assert pointD.batchSize == -3

# def test_crossover():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
#     acts = [ReLu(), ReLu(), Sigmoid(), Sigmoid()]
#     hlc = [2, 1, 2]
#     pointA = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011010)
#     np.random.seed(10011010)
#     co = SomeCrossoverOperator()
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert pointC.input_size == 2
#     assert pointC.output_size == 1
#
#     assert len(pointC.hidden_neuron_counts) == 3
#     assert pointC.hidden_neuron_counts[0] == 2
#     assert pointC.hidden_neuron_counts[1] == 1
#     assert pointC.hidden_neuron_counts[2] == 2
#
#     assert len(pointC.weights) == 4
#     assert np.all(np.isclose(pointC.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[1], np.array([[5, 6.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[2], np.array([[7], [8.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[3], np.array([[-1.51590294, -1.07834955]]), atol=1e-3))
#
#     assert len(pointC.biases) == 4
#     assert np.all(np.isclose(pointC.biases[0], np.array([[-1], [-2]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[1], np.array([[-3]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[2], np.array([[-4], [-5]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[3], np.array([[0]]), atol=1e-3))
#
#     assert len(pointC.activation_functions) == 4
#     assert pointC.activation_functions[0].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[1].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[2].to_string() == Sigmoid().to_string()
#     assert pointC.activation_functions[3].to_string() == Softmax().to_string()
#
#
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     assert pointD.input_size == 2
#     assert pointD.output_size == 1
#
#     assert len(pointD.hidden_neuron_counts) == 1
#     assert pointD.hidden_neuron_counts[0] == 3
#
#     assert len(pointD.weights) == 2
#     assert np.all(np.isclose(pointD.weights[0], np.array([[10, 20], [30, 40.0], [50, 60]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.weights[1], np.array([[0.73125696, 0.60992576, -1.12680355]]), atol=1e-3))
#
#     assert len(pointD.biases) == 2
#     assert np.all(np.isclose(pointD.biases[0], np.array([[-10], [-20.], [-30]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.biases[1], np.array([[0]]), atol=1e-3))
#
#     assert len(pointD.activation_functions) == 2
#     assert pointD.activation_functions[0].to_string() == TanH().to_string()
#     assert pointD.activation_functions[1].to_string() == Sigmoid().to_string()
#
# def test_minimal_damage_crossover():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
#     acts = [ReLu(), ReLu(), Sigmoid(), Sigmoid()]
#     hlc = [2, 1, 2]
#     pointA = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011010)
#     np.random.seed(10011010)
#     co = MinimalDamageCrossoverOperator()
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert pointC.input_size == 2
#     assert pointC.output_size == 1
#
#     assert len(pointC.hidden_neuron_counts) == 3
#     assert pointC.hidden_neuron_counts[0] == 2
#     assert pointC.hidden_neuron_counts[1] == 1
#     assert pointC.hidden_neuron_counts[2] == 2
#
#     assert len(pointC.weights) == 4
#     assert np.all(np.isclose(pointC.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[1], np.array([[5, 6.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[2], np.array([[7], [8.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[3], np.array([[90, 100.0]]), atol=1e-3))
#
#     assert len(pointC.biases) == 4
#     assert np.all(np.isclose(pointC.biases[0], np.array([[-1], [-2]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[1], np.array([[-3]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[2], np.array([[-4], [-5]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[3], np.array([[-60]]), atol=1e-3))
#
#     assert len(pointC.activation_functions) == 4
#     assert pointC.activation_functions[0].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[1].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[2].to_string() == Sigmoid().to_string()
#     assert pointC.activation_functions[3].to_string() == Softmax().to_string()
#
#
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     assert pointD.input_size == 2
#     assert pointD.output_size == 1
#
#     assert len(pointD.hidden_neuron_counts) == 1
#     assert pointD.hidden_neuron_counts[0] == 3
#
#     assert len(pointD.weights) == 2
#     assert np.all(np.isclose(pointD.weights[0], np.array([[10, 20], [30, 40.0], [50, 60]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.weights[1], np.array([[9, 10, 0.0]]), atol=1e-3))
#
#     assert len(pointD.biases) == 2
#     assert np.all(np.isclose(pointD.biases[0], np.array([[-10], [-20.], [-30]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.biases[1], np.array([[-6]]), atol=1e-3))
#
#     assert len(pointD.activation_functions) == 2
#     assert pointD.activation_functions[0].to_string() == TanH().to_string()
#     assert pointD.activation_functions[1].to_string() == Sigmoid().to_string()
#
# def test_layer_resize():
#    layer = resize_layer((3, 3), [-1, 2, ReLu(), np.array([[1, 2], [3, 4]]), np.array([[-1], [-2]])])
#
#    assert layer[0] == -1
#    assert layer[1] == 3
#    assert layer[2].to_string() == ReLu().to_string()
#    assert np.array_equal(layer[3], np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
#    assert np.array_equal(layer[4], np.array([[-1], [-2], [0]]))
#
# def test_layer_resize2():
#     layer = resize_layer((1, 1), [-1, 2, ReLu(), np.array([[1, 2], [3, 4]]), np.array([[-1], [-2]])])
#
#     assert layer[0] == -1
#     assert layer[1] == 1
#     assert layer[2].to_string() == ReLu().to_string()
#     assert np.array_equal(layer[3], np.array([[1]]))
#     assert np.array_equal(layer[4], np.array([[-1]]))

# random.seed(10011010)
# np.random.seed(10011010)
# print(random.randint(0, 3))
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((1, 3)))
# test_crossover()


# test_minimal_damage_crossover()

random.seed(1001)
print(random.randint(0, 5))
print(random.random())
print(random.random())
print(random.random())
print(random.random())

test_simple_crossover()