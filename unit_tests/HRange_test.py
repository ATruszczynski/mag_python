from ann_point.HyperparameterRange import *

def test_hparam():
    hrange = HyperparameterRange((1, 2), (3, 4), [ReLu(), Sigmoid(), TanH(), Softmax()], [QuadDiff(), CrossEntropy()],
                                 (-1, 0), (-3, -2), (-5, -4))

    assert hrange.hiddenLayerCountMin == 1
    assert hrange.hiddenLayerCountMax == 2
    assert hrange.neuronCountMin == 3
    assert hrange.neuronCountMax == 4

    assert len(hrange.actFunSet) == 4
    assert hrange.actFunSet[0].to_string() == ReLu().to_string()
    assert hrange.actFunSet[1].to_string() == Sigmoid().to_string()
    assert hrange.actFunSet[2].to_string() == TanH().to_string()
    assert hrange.actFunSet[3].to_string() == Softmax().to_string()

    assert len(hrange.lossFunSet) == 2
    assert hrange.lossFunSet[0].to_string() == QuadDiff().to_string()
    assert hrange.lossFunSet[1].to_string() == CrossEntropy().to_string()

    assert hrange.learningRateMin == -1
    assert hrange.learningRateMax == 0

    assert hrange.momentumCoeffMin == -3
    assert hrange.momentumCoeffMax == -2

    assert hrange.batchSizeMin == -5
    assert hrange.batchSizeMax == -4
