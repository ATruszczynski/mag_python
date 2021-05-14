from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *


def test_init_no_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=0, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, seed=1001)
    assert len(network.neuronCounts) == 2
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 3
    assert len(network.actFuns) == 2
    assert network.actFuns[0] is None
    assert isinstance(network.actFuns[1], Softmax)
    assert isinstance(network.lossFun, QuadDiff)
    assert network.learningRate == 10
    assert network.momCoeffL == 100

def test_init_2_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=2, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, seed=1001)

    assert len(network.neuronCounts) == 4
    assert network.neuronCounts[0] == 2
    assert network.neuronCounts[1] == 4
    assert network.neuronCounts[1] == 4
    assert network.neuronCounts[3] == 3
    assert len(network.actFuns) == 4
    assert network.actFuns[0] is None
    assert isinstance(network.actFuns[1], ReLu)
    assert isinstance(network.actFuns[2], ReLu)
    assert isinstance(network.actFuns[3], Softmax)
    assert isinstance(network.lossFun, QuadDiff)
    assert network.learningRate == 10
    assert network.momCoeffL == 100
