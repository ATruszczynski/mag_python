from ann_point.HyperparameterRange import *

def test_hparam():
    hrange = HyperparameterRange((1, 2), (3, 4), (1, 5), (10, 20), [ReLu(), Sigmoid(), TanH(), Softmax()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.61, 0.71),
                                 r_prob=(0.45, 0.75))

    assert hrange.min_init_wei == 1
    assert hrange.max_init_wei == 2
    assert hrange.min_init_bia == 3
    assert hrange.max_init_bia == 4
    assert hrange.min_it == 1
    assert hrange.max_it == 5
    assert hrange.min_hidden == 10
    assert hrange.max_hidden == 20

    assert len(hrange.actFunSet) == 4
    assert hrange.actFunSet[0].to_string() == ReLu().to_string()
    assert hrange.actFunSet[1].to_string() == Sigmoid().to_string()
    assert hrange.actFunSet[2].to_string() == TanH().to_string()
    assert hrange.actFunSet[3].to_string() == Softmax().to_string()

    assert hrange.min_mut_radius == 0
    assert hrange.max_mut_radius == 1
    assert hrange.min_wb_mut_prob == 0.05
    assert hrange.max_wb_mut_prob == 0.1
    assert hrange.min_s_mut_prob == 0.6
    assert hrange.max_s_mut_prob == 0.7
    assert hrange.min_p_mut_prob == 0.4
    assert hrange.max_p_mut_prob == 0.6
    assert hrange.min_c_prob == 0.61
    assert hrange.max_c_prob == 0.71
    assert hrange.min_r_prob == 0.45
    assert hrange.max_r_prob == 0.75
