from ann_point.HyperparameterRange import *

def test_hparam():
    hrange = HyperparameterRange((1, 2), (3, 4), [ReLu(), Sigmoid(), TanH()], 5, 6)

    assert hrange.layerCountMin == 1
    assert hrange.layerCountMax == 2
    assert hrange.neuronCountMin == 3
    assert hrange.neuronCountMax == 4
    assert hrange.actFunSet[0].to_string() == ReLu().to_string()
    assert hrange.actFunSet[1].to_string() == Sigmoid().to_string()
    assert hrange.actFunSet[2].to_string() == TanH().to_string()
    assert hrange.weiAbs == 5
    assert hrange.biaAbs == 6
