import pytest
from ann_point.Functions import *

allLossFun = [QuadDiff(), MeanDiff(), CrossEntropy(), ChebyshevLoss(), QuasiCrossEntropy()]

# TODO - A - TEST!!!
def test_quad_diff():
    res = np.array([[1], [0], [1]])
    corr = np.array([[1], [2], [0]])

    qd = QuadDiff()

    assert qd.compute(res, corr) == 5/3
    assert np.array_equal(qd.computeDer(res, corr), np.array([[0], [-4], [2]]) / 3)
    assert qd.to_string() == "QD"
    assert isinstance(qd.copy(), QuadDiff)


def test_mean_diff():
    res = np.array([[-1], [0], [1], [2]])
    corr = np.array([[1], [1], [1], [1]])

    mae = MeanDiff()

    assert mae.compute(res, corr) == 1.0
    assert np.array_equal(mae.computeDer(res, corr), np.array([[-1], [-1], [0], [1]]) / 4)
    assert mae.to_string() == "MAE"
    assert isinstance(mae.copy(), MeanDiff)


def test_cross_entropy():
    res = np.array([[0.2], [0.5], [0.3]])
    corr = np.array([[1], [0.5], [0.2]])

    ce = CrossEntropy()

    ress = ce.compute(res=res, corr=corr)
    assert ress == pytest.approx(3.169321214)
    assert np.all(np.isclose(ce.computeDer(res=res, corr=corr), np.array([[-5], [-1], [-0.66667]])))
    assert ce.to_string() == "CE"
    assert isinstance(ce.copy(), CrossEntropy)


def test_chebyshev():
    res = np.array([[-1], [0], [1], [2]])
    corr = np.array([[1], [1], [1], [1]])

    cl = ChebyshevLoss()

    assert cl.compute(res=res, corr=corr) == pytest.approx(2)
    assert np.all(np.isclose(cl.computeDer(res=res, corr=corr), np.array([[-1], [0], [0], [0]])))
    assert cl.to_string() == "CL"
    assert isinstance(cl.copy(), ChebyshevLoss)

def test_quasi_cross_entropy():
    res = np.array([[-1], [0], [1], [2]])
    corr = np.array([[1], [1], [1], [1]])

    cl = QuasiCrossEntropy()

    assert cl.compute(res=res, corr=corr) == pytest.approx(4)
    # assert np.all(np.isclose(cl.computeDer(res=res, corr=corr), np.array([[-1], [0], [0], [0]])))
    assert cl.to_string() == "QCE"
    assert isinstance(cl.copy(), QuasiCrossEntropy)

def test_non_rep_strings():
    for i in range(len(allLossFun)):
        for j in range(i + 1, len(allLossFun)):
            lf1 = allLossFun[i]
            lf2 = allLossFun[j]
            assert lf1.to_string() != lf2.to_string()

# test_quasi_cross_entropy()
# test_cross_entropy()
# test_quasi_cross_entropy()



