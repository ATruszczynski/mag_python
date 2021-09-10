from utility.Utility import generate_counting_problem, generate_counting_problem_unique
import numpy as np


def test_generate_counting_problem():
    n = 100
    c = 6

    x, y = generate_counting_problem(n, c)

    assert len(x) == n
    assert len(y) == n

    for i in range(n):
        assert x[i].shape[0] == c
        assert x[i].shape[1] == 1
        assert y[i].shape[0] == c + 1
        assert y[i].shape[1] == 1

        ocw = np.where(y[i] == 1)
        assert len(ocw[0]) == 1

        oc = ocw[0][0]

        assert oc == np.sum(x[i])

def test_generate_counting_problem_unique():
    x, y = generate_counting_problem_unique(4)

    assert len(x) == 16
    assert len(y) == 16
    for i in range(len(x)):
        assert x[i].shape[0] == 4
        assert x[i].shape[1] == 1
        assert y[i].shape[0] == 5
        assert y[i].shape[1] == 1

    assert np.array_equal(x[ 0], np.array([[0], [0], [0], [0]]))
    assert np.array_equal(x[ 1], np.array([[1], [0], [0], [0]]))
    assert np.array_equal(x[ 2], np.array([[0], [1], [0], [0]]))
    assert np.array_equal(x[ 3], np.array([[0], [0], [1], [0]]))
    assert np.array_equal(x[ 4], np.array([[0], [0], [0], [1]]))
    assert np.array_equal(x[ 5], np.array([[1], [1], [0], [0]]))
    assert np.array_equal(x[ 6], np.array([[1], [0], [1], [0]]))
    assert np.array_equal(x[ 7], np.array([[1], [0], [0], [1]]))
    assert np.array_equal(x[ 8], np.array([[0], [1], [1], [0]]))
    assert np.array_equal(x[ 9], np.array([[0], [1], [0], [1]]))
    assert np.array_equal(x[10], np.array([[0], [0], [1], [1]]))
    assert np.array_equal(x[11], np.array([[1], [1], [1], [0]]))
    assert np.array_equal(x[12], np.array([[1], [1], [0], [1]]))
    assert np.array_equal(x[13], np.array([[1], [0], [1], [1]]))
    assert np.array_equal(x[14], np.array([[0], [1], [1], [1]]))
    assert np.array_equal(x[15], np.array([[1], [1], [1], [1]]))

    assert np.array_equal(y[ 0], np.array([[1], [0], [0], [0], [0]]))
    assert np.array_equal(y[ 1], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(y[ 2], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(y[ 3], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(y[ 4], np.array([[0], [1], [0], [0], [0]]))
    assert np.array_equal(y[ 5], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[ 6], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[ 7], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[ 8], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[ 9], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[10], np.array([[0], [0], [1], [0], [0]]))
    assert np.array_equal(y[11], np.array([[0], [0], [0], [1], [0]]))
    assert np.array_equal(y[12], np.array([[0], [0], [0], [1], [0]]))
    assert np.array_equal(y[13], np.array([[0], [0], [0], [1], [0]]))
    assert np.array_equal(y[14], np.array([[0], [0], [0], [1], [0]]))
    assert np.array_equal(y[15], np.array([[0], [0], [0], [0], [1]]))
