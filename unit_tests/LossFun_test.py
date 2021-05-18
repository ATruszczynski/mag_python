import pytest
from ann_point.Functions import *

def test_quad_diff():
    res = np.array([[1], [0], [1]])
    corr = np.array([[1], [2], [0]])

    qd = QuadDiff()

    assert qd.compute(res, corr) == 5

def test_quad_diff_der():
    res = np.array([[1], [0], [1]])
    corr = np.array([[1], [2], [0]])

    qd = QuadDiff()

    assert np.array_equal(qd.computeDer(res, corr), np.array([[0], [-4], [2]]))

def test_cross_entropy():
    res = np.array([[0.2], [0.5], [0.3]])
    corr = np.array([[1], [0.5], [0.2]])

    ce = CrossEntropy()
    result = ce.compute(res=res, corr=corr)

    assert result == pytest.approx(2.196806064)

def test_cross_entropy_der():
    res = np.array([[0.2], [0.5], [0.3]])
    corr = np.array([[1], [0.5], [0.2]])

    ce = CrossEntropy()
    result = ce.computeDer(res=res, corr=corr)

    assert np.all(np.isclose(result, np.array([[-5], [-1], [-0.66667]])))


