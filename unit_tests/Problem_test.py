from utility.Utility import generate_counting_problem
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


test_generate_counting_problem()