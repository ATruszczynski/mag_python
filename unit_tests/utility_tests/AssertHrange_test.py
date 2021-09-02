from ann_point.Functions import Softmax, Identity, Poly2, ReLu
from ann_point.HyperparameterRange import assert_hranges_same
from utility.Utility import get_testing_hrange

def test_same():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)


def test_same_aggr():
    hrange = get_testing_hrange()
    hrange.aggrFuns = [Identity(), ReLu()]
    hrange2 = get_testing_hrange()
    hrange2.aggrFuns = [Identity(), ReLu()]

    assert_hranges_same(hrange, hrange2)


def test_diff_min_init_wei():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_init_wei = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_init_wei():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_init_wei = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_init_bia():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_init_bia = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_init_bia():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_init_bia = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_it():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_it = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_it():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_it = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_hidden():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_hidden = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_hidden():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_hidden = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_actFunSet_shorter():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.actFunSet = hrange2.actFunSet[:3]

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_actFunSet_longer():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.actFunSet.append(Softmax())

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_actFunSet_different():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.actFunSet[0] = Identity()

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_mut_radius():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_mut_radius = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_mut_radius():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_mut_radius = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_sqr_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_swap = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_sqr_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_swap = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_lin_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_multi = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_lin_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_multi = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_p_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_p_prob = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_p_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_p_prob = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_c_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_c_prob = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_c_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_c_prob = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_min_dstr_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.min_p_rad = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_max_dstr_mut_prob():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.max_p_rad = -100

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False

def test_nonNone_aggr():
    hrange = get_testing_hrange()
    hrange2 = get_testing_hrange()

    assert_hranges_same(hrange, hrange2)

    hrange2.aggrFuns = [Identity()]

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False

def test_diff_aggr():
    hrange = get_testing_hrange()
    hrange.aggrFuns = [ReLu(), Poly2()]
    hrange2 = hrange.copy()

    assert_hranges_same(hrange, hrange2)

    hrange2.aggrFuns = [ReLu(), Identity()]

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_len_aggr():
    hrange = get_testing_hrange()
    hrange.aggrFuns = [ReLu(), Poly2()]
    hrange2 = hrange.copy()

    assert_hranges_same(hrange, hrange2)

    hrange2.aggrFuns = [ReLu(), Poly2(), Identity()]

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False


def test_diff_None_aggr():
    hrange = get_testing_hrange()
    hrange.aggrFuns = [ReLu(), Poly2()]
    hrange2 = hrange.copy()

    assert_hranges_same(hrange, hrange2)

    hrange2.aggrFuns = [ReLu(), None]

    try:
        assert_hranges_same(hrange, hrange2)
    except AssertionError:
        assert True
    else:
        assert False