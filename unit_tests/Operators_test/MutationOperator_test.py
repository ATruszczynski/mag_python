import random

import numpy as np
import pytest

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import *
from utility.Mut_Utility import *

def test_simple_mutation():
    #TODO fix with it changes
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()])
    mo = SimpleCNMutation(hrange)

    random.seed(1001)
    np.random.seed(1001)

    link1 = np.array([[0, 1, 1, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 4],
                      [0, 0, 3, 0, 5],
                      [0, 7, 0, 0, 6],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    bia1 = np.array([[-1., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), Sigmoid(), Sigmoid()]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(), actFuns=actFuns1, aggrFun=TanH(), maxit=2)
    # cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1, aggrFun=TanH())

    mutant = mo.mutate(cn1, pm=0.75, radius=1)

    assert np.array_equal(mutant.links, link1)
    assert np.all(np.isclose(mutant.weights, np.array([[0, 1.54176999, 1.47983043, 0, 5.20238865],
                                                       [0, 0, 2.875572, 0, 5.43418561],
                                                       [0, 5.39956767, 0, 0, 5.97519725],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(mutant.bias, np.array([[-1, -1.14810728, -3.75488531, -6.28762932, -4.89076131]]), atol=1e-5))
    assert mutant.actFuns[0] is None
    assert mutant.actFuns[1].to_string() == ReLu().to_string()
    assert mutant.actFuns[2].to_string() == ReLu().to_string()
    assert mutant.actFuns[3].to_string() == Sigmoid().to_string()
    assert mutant.actFuns[4].to_string() == Sigmoid().to_string()
    assert mutant.aggrFun.to_string() == TanH().to_string()

    assert mutant.hidden_comp_order is None

    assert mutant.aggrFun.to_string() == TanH().to_string()

    assert mutant.maxit == 2

def test_struct_mutation():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()])
    mo = SimpleAndStructuralCNMutation(hrange, 2)

    random.seed(1001)
    np.random.seed(1001)

    link1 = np.array([[0, 1, 1, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 4],
                     [0, 0, 3, 0, 5],
                     [0, 7, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(), actFuns=actFuns1, aggrFun=TanH(), maxit=2)

    mutant = mo.mutate(cn1, pm=0.75, radius=1)

    assert np.array_equal(mutant.links, np.array([[0, 0, 0, 1, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]))
    assert np.all(np.isclose(mutant.weights, np.array([[0, 0, 0, 1.06135843, 0],
                                                       [0, 0, 2.875572, 3.00067462, 0],
                                                       [0, 0, 0, 3.67703328, 0],
                                                       [0, 0, 0, 0, 0],
                                                       [0, 0, 0, 0, 0]]), atol=1e-5))
    assert np.all(np.isclose(mutant.bias, np.array([[0, -1.14810728, -3.75488531, -6.28762932, -4.89076131]]), atol=1e-5))
    assert len(mutant.actFuns) == 5
    assert mutant.actFuns[0] is None
    assert mutant.actFuns[1].to_string() == Sigmoid().to_string()
    assert mutant.actFuns[2].to_string() == ReLu().to_string()
    assert mutant.actFuns[3] is None
    assert mutant.actFuns[4] is None
    assert mutant.aggrFun.to_string() == Sigmoid().to_string()

    assert mutant.hidden_comp_order is None

    assert mutant.maxit == 2





seed = 1001
random.seed(seed)
np.random.seed(seed)
wei1 = np.array([[0., 1, 2, 0, 4],
                 [0, 0, 3, 0, 5],
                 [0, 7, 0, 0, 6],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
link1 = np.array([[0, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
bia1 = np.array([[-1, -2, -3, -4, -5]])
ch1 = np.zeros((5, 5))
ch1[np.random.random((5, 5)) <= 0.75] = 1
ch1[:, 0] = 0
ch1[3:5, :] = 0
np.fill_diagonal(ch1, 0)
# print(f"change_prob: \n {ch1}")
weights = np.random.normal(wei1, 1, (5, 5))
weights = np.multiply(weights, ch1)
weights = np.multiply(weights, link1)
# print(f"weights: \n {wei_moves}")
ch1b = np.zeros((1, 5))
ch1b[np.random.random((1, 5)) <= 0.75] = 1
biases = np.random.normal(bia1, 1, (1, 5))
biases = np.multiply(biases, ch1b)
ch_it = random.random()
if ch_it <= 0.75:
    print(f"change_it: \n{random.randint(1, 5)}")
ch_l = np.zeros(wei1.shape)
ch_l[np.random.random(wei1.shape) <= 0.75] = 1
ch_l[:, 0] = 0
ch_l[3:5, :] = 0
np.fill_diagonal(ch_l, 0)
# print(f"ch_l: \n{ch_l}")
links = link1.copy()
links[ch_l == 1] = 1 - links[ch_l == 1]
# print(f"lc: \n{link2}")
edge_status = link1 - links
# print(f"status_edges: \n{edge_status}")
minW = np.min(weights)
maxW = np.max(weights)
new_edges_weights = weights.copy()
added_edges = np.where(edge_status == -1)
new_edges_weights[added_edges] = np.random.uniform(minW, maxW, (5, 5))[added_edges]
new_edges_weights = np.multiply(new_edges_weights, links)
weights = new_edges_weights
ca_1 = random.random()
if ca_1 <= 0.75:
    print(f"ca_1: {try_choose_different(ReLu(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
ca_2 = random.random()
if ca_2 <= 0.75:
    print(f"ca_2: {try_choose_different(ReLu(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
aggc = random.random()
if aggc <= 0.75:
    print(f"aggr: {try_choose_different(TanH(), [ReLu(), Sigmoid(), GaussAct(), TanH()])}")
hc_c = random.random()
if hc_c <= 0.75:
    new_hc = try_choose_different(2, [0, 1, 2, 3, 4])
    print(f"new hc: {new_hc}")
    if new_hc < 2:
        to_preserve = [0, 1, 2, 3, 4]
        brr = choose_without_repetition([1, 2], 2 - new_hc)
        for i in brr:
            to_preserve.remove(i)
        to_preserve = np.array(to_preserve).reshape(1, -1)

        links = links[to_preserve[0, :, None], to_preserve]
        weights = weights[to_preserve[0, :, None], to_preserve]
        new_biases = biases[0, to_preserve]
        new_af = []
        print(f"to_preserve: {to_preserve}")

    elif new_hc > 2:
        act_set = [ReLu(), Sigmoid(), GaussAct(), TanH()]
        link3 = get_links(1, 2, 1 + 2 + new_hc)
        link3[:1 + 2, :1 + 2] = links[:1 + 2, :1 + 2]
        link3[:1 + 2, -2:] = links[:1 + 2, -2:]

        links = link3

        minW = np.min(weights)
        maxW = np.max(weights)
        weight3 = np.random.uniform(minW, maxW, (1 + 2 + new_hc, 1 + 2 + new_hc))
        weight3[:1 + 2, :1 + 2] = weights[:1 + 2, :1 + 2]
        weight3[:1 + 2, -2:] = weights[:1 + 2, -2:]
        weight3 = np.multiply(weight3, link3)

        weights = weight3

        minB = np.min(biases)
        maxB = np.max(biases)
        bia3 = np.random.uniform(minB, maxB, (1, 1 + 2 + new_hc))
        bia3[0, :1 + 2] = biases[0, :1 + 2]
        bia3[0, -2:] = biases[0, -2:]

        biases = bia3

        for i in range(new_hc - 2):
            print(f"inc_add_{i+1}: {act_set[random.randint(0, len(act_set) - 1)]}")


print(f"new_links: \n {links}")
print(f"new_weights: \n{weights}")
print(f"biases: \n {biases}")

test_struct_mutation()

