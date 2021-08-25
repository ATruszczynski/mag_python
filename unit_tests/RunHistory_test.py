import random

import pytest

from utility.CNDataPoint import CNDataPoint
from utility.RunHistory import RunHistory
from utility.TestingUtility import assert_chaos_networks_same
from utility.Utility import generate_population, get_default_hrange_ga, choose_without_repetition
import numpy as np


def test_add_iteration():
    random.seed(1001)
    np.random.seed(1001)

    hrange = get_default_hrange_ga()
    nets = generate_population(hrange, 5, 2, 3)
    cndps_p = []
    for i in range(len(nets)):
        cndp = CNDataPoint(nets[i])
        cndp.add_data(i * 0.1, np.array([[i, 2-i], [i**2, 0]]))
        cndps_p.append(cndp)

    random.seed(1001)
    np.random.seed(1001)
    order = choose_without_repetition([0, 1, 2, 3, 4], 5)
    cndps = []
    for i in range(len(order)):
        cndps.append(cndps_p[order[i]])

    rh = RunHistory()

    rh.add_it_hist(cndps_p)
    rh.add_it_hist(cndps)

    assert len(rh.it_hist) == 2

    assert len(rh.it_hist[0]) == 5
    assert len(rh.it_hist[1]) == 5

    cndps[4].ff = 333
    cndps_p[3].conf_mat = np.zeros((1, 1))

    # rh.it_hist[0] test
    assert_chaos_networks_same(rh.it_hist[0][0].net, nets[0])
    assert rh.it_hist[0][0].ff == pytest.approx(0)
    assert np.array_equal(rh.it_hist[0][0].conf_mat, np.array([[0, 2], [0, 0]]))

    assert_chaos_networks_same(rh.it_hist[0][1].net, nets[1])
    assert rh.it_hist[0][1].ff == pytest.approx(0.1)
    assert np.array_equal(rh.it_hist[0][1].conf_mat, np.array([[1, 1], [1, 0]]))

    assert_chaos_networks_same(rh.it_hist[0][2].net, nets[2])
    assert rh.it_hist[0][2].ff == pytest.approx(0.2)
    assert np.array_equal(rh.it_hist[0][2].conf_mat, np.array([[2, 0], [4, 0]]))

    assert_chaos_networks_same(rh.it_hist[0][3].net, nets[3])
    assert rh.it_hist[0][3].ff == pytest.approx(0.3)
    assert np.array_equal(rh.it_hist[0][3].conf_mat, np.array([[3, -1], [9, 0]]))

    assert_chaos_networks_same(rh.it_hist[0][4].net, nets[4])
    assert rh.it_hist[0][4].ff == pytest.approx(0.4)
    assert np.array_equal(rh.it_hist[0][4].conf_mat, np.array([[4, -2], [16, 0]]))

    # rh.it_hist[1] test
    assert_chaos_networks_same(rh.it_hist[1][0].net, nets[0])
    assert rh.it_hist[1][0].ff == pytest.approx(0)
    assert np.array_equal(rh.it_hist[1][0].conf_mat, np.array([[0, 2], [0, 0]]))

    assert_chaos_networks_same(rh.it_hist[1][1].net, nets[2])
    assert rh.it_hist[1][1].ff == pytest.approx(0.2)
    assert np.array_equal(rh.it_hist[1][1].conf_mat, np.array([[2, 0], [4, 0]]))

    assert_chaos_networks_same(rh.it_hist[1][2].net, nets[1])
    assert rh.it_hist[1][2].ff == pytest.approx(0.1)
    assert np.array_equal(rh.it_hist[1][2].conf_mat, np.array([[1, 1], [1, 0]]))

    assert_chaos_networks_same(rh.it_hist[1][3].net, nets[4])
    assert rh.it_hist[1][3].ff == pytest.approx(0.4)
    assert np.array_equal(rh.it_hist[1][3].conf_mat, np.array([[4, -2], [16, 0]]))

    assert_chaos_networks_same(rh.it_hist[1][4].net, nets[3])
    assert rh.it_hist[1][4].ff == pytest.approx(0.3)
    assert np.array_equal(rh.it_hist[1][4].conf_mat, np.array([[3, -1], [9, 0]]))

    cndps_p[0].ff = 111
    assert cndps_p[0].ff == 111
    assert cndps[0].ff == 111
    assert rh.it_hist[0][0].ff == pytest.approx(0)
    assert rh.it_hist[1][0].ff == pytest.approx(0)

    rh.it_hist[0][1].ff = 222
    assert rh.it_hist[0][1].ff == pytest.approx(222)
    assert rh.it_hist[1][2].ff == pytest.approx(0.1)

random.seed(1001)
np.random.seed(1001)
order = choose_without_repetition([0, 1, 2, 3, 4], 5)
print(order)
test_add_iteration()
