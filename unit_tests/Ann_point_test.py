from ann_point.AnnPoint import *
from ann_point.Functions import *

def test_ann_point_constructor():
    ann = AnnPoint(neuronCounts=[1, 4, 4, 4, 2], actFuns=[ReLu(), ReLu(), TanH(), Sigmoid()],
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    assert len(ann.neuronCounts) == 5
    assert ann.neuronCounts[0] == 1
    assert ann.neuronCounts[1] == 4
    assert ann.neuronCounts[2] == 4
    assert ann.neuronCounts[3] == 4
    assert ann.neuronCounts[4] == 2

    assert len(ann.actFuns) == 4
    assert ann.actFuns[0].to_string() == ReLu().to_string()
    assert ann.actFuns[1].to_string() == ReLu().to_string()
    assert ann.actFuns[2].to_string() == TanH().to_string()
    assert ann.actFuns[3].to_string() == Sigmoid().to_string()

    assert ann.lossFun.to_string() == QuadDiff().to_string()

    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7

def test_ann_point_copy():
    ann = AnnPoint(neuronCounts=[1, 4, 2], actFuns=[ReLu(), Softmax()],
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    assert len(ann.neuronCounts) == 3
    assert ann.neuronCounts[0] == 1
    assert ann.neuronCounts[1] == 4
    assert ann.neuronCounts[2] == 2

    assert len(ann.actFuns) == 2
    assert ann.actFuns[0].to_string() == ReLu().to_string()
    assert ann.actFuns[1].to_string() == Softmax().to_string()

    assert ann.lossFun.to_string() == QuadDiff().to_string()

    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7


    ann2 = ann.copy()

    assert len(ann2.neuronCounts) == 3
    assert ann2.neuronCounts[0] == 1
    assert ann2.neuronCounts[1] == 4
    assert ann2.neuronCounts[2] == 2

    assert len(ann2.actFuns) == 2
    assert ann2.actFuns[0].to_string() == ReLu().to_string()
    assert ann2.actFuns[1].to_string() == Softmax().to_string()

    assert ann2.lossFun.to_string() == QuadDiff().to_string()

    assert ann2.learningRate == 5
    assert ann2.momCoeff == 6
    assert ann2.batchSize == 7




    ann2.neuronCounts[1] = 30
    ann2.actFuns[0] = TanH()
    ann2.actFuns[1] = Sigmoid()
    ann2.lossFun = CrossEntropy()
    ann2.learningRate = 50
    ann2.momCoeff = 60
    ann2.batchSize = 70

    ann.neuronCounts[1] = 100
    ann.lossFun = CrossEntropy()

    assert len(ann.neuronCounts) == 3
    assert ann.neuronCounts[0] == 1
    assert ann.neuronCounts[1] == 100
    assert ann.neuronCounts[2] == 2

    assert len(ann.actFuns) == 2
    assert ann.actFuns[0].to_string() == ReLu().to_string()
    assert ann.actFuns[1].to_string() == Softmax().to_string()

    assert ann.lossFun.to_string() == CrossEntropy().to_string()

    assert ann.learningRate == 5
    assert ann.momCoeff == 6
    assert ann.batchSize == 7





    assert len(ann2.neuronCounts) == 3
    assert ann2.neuronCounts[0] == 1
    assert ann2.neuronCounts[1] == 30
    assert ann2.neuronCounts[2] == 2

    assert len(ann2.actFuns) == 2
    assert ann2.actFuns[0].to_string() == TanH().to_string()
    assert ann2.actFuns[1].to_string() == Sigmoid().to_string()

    assert ann2.lossFun.to_string() == CrossEntropy().to_string()

    assert ann2.learningRate == 50
    assert ann2.momCoeff == 60
    assert ann2.batchSize == 70

def test_ann_point_string():
    ann = AnnPoint(neuronCounts=[1, 2, 3], actFuns=[ReLu(), Softmax()],
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    string = ann.to_string()

    assert string == "|1|RL|2|SM|3|QD|5|6|7|"

def test_ann_point_full_string():
    ann = AnnPoint(neuronCounts=[1, 2, 3], actFuns=[ReLu(), Softmax()],
                   lossFun=QuadDiff(), learningRate=5.2345, momCoeff=6.3456, batchSize=7.4567)

    string = ann.to_string_full()

    assert string == "|1|RL|2|SM|3|QD|5.2345|6.3456|7.4567|"

def test_ann_point_size():
    ann = AnnPoint(neuronCounts=[1, 4, 5, 4, 3], actFuns=[ReLu(), Sigmoid()],
                   lossFun=QuadDiff(), learningRate=5, momCoeff=6, batchSize=7)

    size = ann.size()

    assert size == 4 + 20 + 20 + 12






