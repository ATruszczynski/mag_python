from ann_point.HyperparameterRange import *

def test_hparam():
    hrange = HyperparameterRange((1, 2), (3, 4), [ReLu(), Sigmoid()], [Softmax(), TanH()],
                                 [QuadDiff(), CrossEntropy()], (5, 6), (7, 8), (9, 10))

    assert hrange.layerCountMin == 1
    assert hrange.layerCountMax == 2
    assert hrange.neuronCountMin == 3
    assert hrange.neuronCountMax == 4
    assert hrange.actFunSet[0].to_string() == ReLu().to_string()
    assert hrange.actFunSet[1].to_string() == Sigmoid().to_string()
    assert hrange.aggrFunSet[0].to_string() == Softmax().to_string()
    assert hrange.aggrFunSet[1].to_string() == TanH().to_string()
    assert hrange.lossFunSet[0].to_string() == QuadDiff().to_string()
    assert hrange.lossFunSet[1].to_string() == CrossEntropy().to_string()
    assert hrange.learningRateMin == 5
    assert hrange.learningRateMax == 6
    assert hrange.momentumCoeffMin == 7
    assert hrange.momentumCoeffMax == 8
    assert hrange.batchSizeMin == 9
    assert hrange.batchSizeMax == 10
