from ann_point.AnnPoint import *
from ann_point.Functions import *

def test_ann_point_constructor():
    ann = AnnPoint(inputSize=1, outputSize=2, hiddenLayerCount=3, neuronCount=4, actFun=ReLu(), aggrFun=Sigmoid(),
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    assert ann.inputSize == 1
    assert ann.outputSize == 2
    assert ann.hiddenLayerCount == 3
    assert ann.neuronCount == 4
    assert ann.actFun.to_string() == ReLu().to_string()
    assert ann.aggrFun.to_string() == Sigmoid().to_string()
    assert ann.lossFun.to_string() == QuadDiff().to_string()
    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7

def test_ann_point_copy():
    ann = AnnPoint(inputSize=1, outputSize=2, hiddenLayerCount=3, neuronCount=4, actFun=ReLu(), aggrFun=Sigmoid(),
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    assert ann.inputSize == 1
    assert ann.outputSize == 2
    assert ann.hiddenLayerCount == 3
    assert ann.neuronCount == 4
    assert ann.actFun.to_string() == ReLu().to_string()
    assert ann.aggrFun.to_string() == Sigmoid().to_string()
    assert ann.lossFun.to_string() == QuadDiff().to_string()
    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7

    ann2 = ann.copy()
    ann2.inputSize = 10
    ann2.outputSize = 20
    ann2.hiddenLayerCount = 30
    ann2.neuronCount = 40
    ann2.actFun = Softmax()
    ann2.aggrFun = TanH()
    ann2.lossFun = CrossEntropy()
    ann2.learningRate = 50
    ann2.momCoeff = 60
    ann2.batchSize = 70

    ann.inputSize = 100
    ann.neuronCount = 400
    ann.lossFun = CrossEntropy()

    assert ann.inputSize == 100
    assert ann.outputSize == 2
    assert ann.hiddenLayerCount == 3
    assert ann.neuronCount == 400
    assert ann.actFun.to_string() == ReLu().to_string()
    assert ann.aggrFun.to_string() == Sigmoid().to_string()
    assert ann.lossFun.to_string() == CrossEntropy().to_string()
    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7

    assert ann2.inputSize == 10
    assert ann2.outputSize == 20
    assert ann2.hiddenLayerCount == 30
    assert ann2.neuronCount == 40
    assert ann2.actFun.to_string() == Softmax().to_string()
    assert ann2.aggrFun.to_string() == TanH().to_string()
    assert ann2.lossFun.to_string() == CrossEntropy().to_string()
    assert ann2.learningRate == 50
    assert ann2.momCoeff == 60
    assert ann2.batchSize == 70

def test_ann_point_string():
    ann = AnnPoint(inputSize=1, outputSize=2, hiddenLayerCount=3, neuronCount=4, actFun=ReLu(), aggrFun=Sigmoid(),
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    string = ann.to_string()

    assert string == "|1|2|3|4|RL|SG|QD|5|6|7|"

def test_ann_point_size():
    ann = AnnPoint(inputSize=1, outputSize=2, hiddenLayerCount=3, neuronCount=4, actFun=ReLu(), aggrFun=Sigmoid(),
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    size = ann.size()

    assert size == 16 + 256 + 256 + 32





