from ann_point.HyperparameterRange import *

def test_hparam():
    hrange = HyperparameterRange((1, 2), (3, 4), (1, 5), [ReLu(), Sigmoid(), TanH(), Softmax()])

    assert hrange.min_init_wei == 1
    assert hrange.max_init_wei == 2
    assert hrange.min_init_bia == 3
    assert hrange.max_init_bia == 4
    assert hrange.min_it == 1
    assert hrange.max_it == 5

    assert len(hrange.actFunSet) == 4
    assert hrange.actFunSet[0].to_string() == ReLu().to_string()
    assert hrange.actFunSet[1].to_string() == Sigmoid().to_string()
    assert hrange.actFunSet[2].to_string() == TanH().to_string()
    assert hrange.actFunSet[3].to_string() == Softmax().to_string()
