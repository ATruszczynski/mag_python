from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *


def test_init_no_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=0, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, seed=1001)

def test_init_2_hidden():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=2, neuronCount=2, actFun=ReLu(),
                                       aggrFun=Softmax(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, seed=1001)

    assert network.neuronCounts[2] == 4
