import numpy as np

from utility.CNDataPoint import CNDataPoint
from utility.TestingUtility import assert_chaos_networks_same
from utility.Utility import get_default_hrange_ga, generate_population
from utility.Utility2 import *


def test_cndp_constructor():
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=3)[0]

    cndp1 = CNDataPoint(cn)

    assert cndp1.ff == 0.
    assert cndp1.conf_mat is None
    assert_chaos_networks_same(cndp1.net, cn)

    cn.input_size = 22

    assert cndp1.net.input_size == 2
    assert cn.input_size == 22

def test_add_data():
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=3)[0]

    cndp1 = CNDataPoint(cn)

    cndp1.add_data(0.5, np.array([[1, 2, 0],
                                  [2, 4, 1],
                                  [0, 2, 4]]))

    assert cndp1.ff == 0.5
    assert np.array_equal(cndp1.conf_mat,  np.array([[1, 2, 0],
                                                     [2, 4, 1],
                                                     [0, 2, 4]]))
    assert_chaos_networks_same(cndp1.net, cn)

def test_gets():
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=3)[0]

    cndp1 = CNDataPoint(cn)

    cndp1.add_data(0.5, np.array([[1, 2, 0],
                                  [2, 4, 1],
                                  [0, 2, 4]]))

    assert cndp1.get_acc() == accuracy(np.array([[1, 2, 0],
                                                 [2, 4, 1],
                                                 [0, 2, 4]]))
    assert cndp1.get_avg_prec() == average_precision(np.array([[1, 2, 0],
                                                               [2, 4, 1],
                                                               [0, 2, 4]]))
    assert cndp1.get_avg_rec() == average_recall(np.array([[1, 2, 0],
                                                               [2, 4, 1],
                                                               [0, 2, 4]]))
    assert cndp1.get_avg_f1() == average_f1_score(np.array([[1, 2, 0],
                                                               [2, 4, 1],
                                                               [0, 2, 4]]))
    assert cndp1.get_eff() == efficiency(np.array([[1, 2, 0],
                                                               [2, 4, 1],
                                                               [0, 2, 4]]))
    assert cndp1.get_meff() == m_efficiency(np.array([[1, 2, 0],
                                                      [2, 4, 1],
                                                      [0, 2, 4]]))

# TODO - C - this could be better
def test_copy():
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=3)[0]

    cndp1 = CNDataPoint(cn)

    cndp1.add_data(0.5, np.array([[1, 2, 0],
                                  [2, 4, 1],
                                  [0, 2, 4]]))
    cp = cndp1.copy()

    assert_chaos_networks_same(cndp1.net, cp.net)
    assert cndp1.ff == cp.ff
    assert np.array_equal(cndp1.conf_mat, cp.conf_mat)

    cndp1.net.input_size = 222
    assert cp.net.input_size == 2
    cndp1.ff = -23213123
    assert cp.ff == 0.5
    cndp1.conf_mat[1, 1] = 22222
    assert np.array_equal(cp.conf_mat, np.array([[1, 2, 0],
                                            [2, 4, 1],
                                            [0, 2, 4]]))


def test_copy_2():
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=3)[0]

    cndp1 = CNDataPoint(cn)

    cndp1.add_data(0.5, None)
    cp = cndp1.copy()

    assert_chaos_networks_same(cndp1.net, cp.net)
    assert cndp1.ff == cp.ff
    assert cp.conf_mat is None

    cndp1.net.input_size = 222
    assert cp.net.input_size == 2
    cndp1.ff = -23213123
    cndp1.conf_mat = np.zeros((1, 1))
    assert cp.ff == 0.5
    assert cp.conf_mat is None
test_copy_2()


# test_cndp_constructor()
# test_add_data()
# test_gets()
# test_copy()