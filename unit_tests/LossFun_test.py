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

