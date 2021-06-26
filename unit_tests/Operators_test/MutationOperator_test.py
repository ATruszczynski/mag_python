import random

import numpy as np
import pytest

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import *
from utility.Mut_Utility import *
from utility.TestingUtility import compare_chaos_network


def test_simple_mutation():
    #TODO fix with it changes
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
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

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(), actFuns=actFuns1,
                   aggrFun=TanH(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)
    # cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1, aggrFun=TanH())

    mutant = mo.mutate(cn1, wb_pm=0.75, s_pm=0.75, radius=1)

    compare_chaos_network(mutant,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 1, 0, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 1.54176999, 1.47983043, 0, 5.20238865],
                                                    [0, 0, 2.875572, 0, 5.43418561],
                                                    [0, 5.39956767, 0, 0, 5.97519725],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[-1, -1.14810728, -3.75488531, -6.28762932, -4.89076131]]),
                          desired_actFun=[None, ReLu(), ReLu(), Sigmoid(), Sigmoid()], #TODO ten test nie ma None'ów na końcu
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3)

def test_struct_mutation():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
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

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(), actFuns=actFuns1, aggrFun=TanH(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)

    mutant = mo.mutate(cn1, wb_pm=0.75, s_pm=0.75, radius=1)

    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 1, 0, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0., 1, 2, 0, 4],
                                                    [0, 0, 3, 0, 5],
                                                    [0, 7, 0, 0, 6],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -4, -5]]),
                          desired_actFun=[None, ReLu(), ReLu(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3)

    compare_chaos_network(net=mutant,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 0, 0, 1, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0, 1.06135843, 0],
                                                    [0, 0, 2.875572, 3.00067462, 0],
                                                    [0, 0, 0, 3.67703328, 0],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, -1.14810728, -3.75488531, -6.28762932, -4.89076131]]),
                          desired_actFun=[None, Sigmoid(), ReLu(), None, None],
                          desired_aggr=Sigmoid(),
                          desired_maxit=2,
                          desired_mut_rad=0.37447,
                          desired_wb_prob=0.070046,
                          desired_s_prob=0.6607793)

def test_struct_mutation_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
    mo = SimpleAndStructuralCNMutation(hrange, 2)

    random.seed(1003)
    np.random.seed(1003)

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

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1, aggrFun=TanH(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)

    mutant = mo.mutate(cn1, wb_pm=0.75, s_pm=0.75, radius=1)


    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 1, 0, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0., 1, 2, 0, 4],
                                                    [0, 0, 3, 0, 5],
                                                    [0, 7, 0, 0, 6],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -4, -5]]),
                          desired_actFun=[None, ReLu(), ReLu(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3)

    compare_chaos_network(net=mutant,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=5,
                          desired_hidden_count=4,
                          desired_links=np.array([[0., 1., 0., 1., 0., 1., 0.],
                                                  [0., 0., 1., 0., 0., 1., 0.],
                                                  [0., 0., 0., 0., 0., 0., 1.],
                                                  [0., 0., 1., 0., 1., 0., 0.],
                                                  [0., 0., 1., 0., 0., 1., 1.],
                                                  [0., 0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0., 0.]]),
                          desired_weights=np.array([[-0.,0.62183713,  0.,          0.02160549,  0.,          4.87977554, 0.        ],
                                                    [ 0.,0.,          3.,          0.,          0.,          5.12118995, 0.        ],
                                                    [-0.,0.,         -0.,          0.,          0.,         -0.,         5.67832716],
                                                    [ 0.,0.,          0.99854234,  0.,          4.3555107,   0.,         0.,        ],
                                                    [ 0.,0.,          2.04770976,  0.,          0.,          1.4909221,  2.79087538],
                                                    [ 0.,0.,          0.,          0.,          0.,          0.,         0.        ],
                                                    [ 0.,0.,          0.,          0.,          0.,          0.,         0.        ]]),
                          desired_biases=np.array([[0., -2., -1.42185604, -3.89839703, -4.75626278, -4., -5.]]),
                          desired_actFun=[None, GaussAct(), GaussAct(), ReLu(), GaussAct(), None, None],
                          desired_aggr=GaussAct(),
                          desired_maxit=4,
                          desired_mut_rad=1,
                          desired_wb_prob=0.075295,
                          desired_s_prob=0.669351)



def test_struct_mutation_3():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
    mo = SimpleAndStructuralCNMutation(hrange, 2)

    random.seed(1004)
    np.random.seed(1004)

    link1 = np.array([[0, 0, 1, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 0, 1, 4, 6],
                     [0, 0, 2, 0, 7],
                     [0, 0, 0, 5, 8],
                     [0, 0, 3, 0, 9],
                     [0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., 0, -3, -4, -5]])
    actFuns1 = [None, None, ReLu(), ReLu(), None]

    cn1 = ChaosNet(input_size=2, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(), actFuns=actFuns1, aggrFun=TanH(), maxit=4, mutation_radius=1,
                   wb_mutation_prob=2, s_mutation_prob=3)

    mutant = mo.mutate(cn1, wb_pm=0.75, s_pm=0.75, radius=1)


    # assert np.array_equal(cn1.links, np.array([[0, 0, 1, 1, 1],
    #                                            [0, 0, 1, 0, 1],
    #                                            [0, 0, 0, 1, 1],
    #                                            [0, 0, 1, 0, 1],
    #                                            [0, 0, 0, 0, 0]]))
    # assert np.all(np.isclose(cn1.weights, np.array([[0, 0, 1, 4, 6],
    #                                                    [0, 0, 2, 0, 7],
    #                                                    [0, 0, 0, 5, 8],
    #                                                    [0, 0, 3, 0, 9],
    #                                                    [0, 0, 0, 0, 0.]]), atol=1e-5))
    # assert np.all(np.isclose(cn1.biases, np.array([[0., 0, -3, -4, -5]]), atol=1e-5))
    # assert len(cn1.actFuns) == 5
    # assert cn1.actFuns[0] is None
    # assert cn1.actFuns[1] is None
    # assert cn1.actFuns[2].to_string() == ReLu().to_string()
    # assert cn1.actFuns[3].to_string() == ReLu().to_string()
    # assert cn1.actFuns[4] is None
    # assert cn1.aggrFun.to_string() == TanH().to_string()
    #
    # assert cn1.hidden_comp_order is None

    compare_chaos_network(net=cn1,
                          desired_input_size=2,
                          desited_output_size=1,
                          desired_neuron_count=5,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=4,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 0, 1, 1, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 0, 0, 1, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 1, 4, 6],
                                                    [0, 0, 2, 0, 7],
                                                    [0, 0, 0, 5, 8],
                                                    [0, 0, 3, 0, 9],
                                                    [0, 0, 0, 0, 0.]]),
                          desired_biases=np.array([[0., 0, -3, -4, -5]]),
                          desired_actFun=[None, None, ReLu(), ReLu(), None],
                          desired_aggr=TanH(),
                          desired_maxit=4,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3)

    #
    # assert np.array_equal(mutant.links, np.array([[0, 0, 0],
    #                                               [0, 0, 1],
    #                                               [0, 0, 0]]))
    # assert np.all(np.isclose(mutant.weights, np.array([[0, 0, 0],
    #                                                    [0, 0, 6.2377573],
    #                                                    [0, 0, 0]]), atol=1e-5))
    # assert np.all(np.isclose(mutant.biases, np.array([[0., 0., -5.]]), atol=1e-5))
    # assert len(mutant.actFuns) == 3
    # assert mutant.actFuns[0] is None
    # assert mutant.actFuns[1] is None
    # assert mutant.actFuns[2] is None
    # assert mutant.aggrFun.to_string() == TanH().to_string()
    #
    # assert mutant.hidden_comp_order is None

    compare_chaos_network(net=mutant,
                          desired_input_size=2,
                          desited_output_size=1,
                          desired_neuron_count=3,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=2,
                          desired_hidden_count=0,
                          desired_links=np.array([[0, 0, 0],
                                                  [0, 0, 1],
                                                  [0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0],
                                                    [0, 0, 6.2377573],
                                                    [0, 0, 0]]),
                          desired_biases=np.array([[0., 0, -5]]),
                          desired_actFun=[None, None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=0.235941,
                          desired_wb_prob=0.0857273,
                          desired_s_prob=0.609284)

seed = 1001
random.seed(seed)
np.random.seed(seed)
wei1 = np.array([[0, 0, 1, 4, 6],
                 [0, 0, 2, 0, 7],
                 [0, 0, 0, 5, 8],
                 [0, 0, 3, 0, 9],
                 [0, 0, 0, 0, 0.]])
link1 = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0]])
bia1 = np.array([[0., 0, -3, -4, -5]])
maxit = 4
input_size = 2
hidden_size = 2
output_size = 1
neuron_count = 5


ch1 = np.zeros((neuron_count, neuron_count))
ch1[np.random.random((neuron_count, neuron_count)) <= 0.75] = 1
ch1[:, :input_size] = 0
ch1[-output_size:, :] = 0
np.fill_diagonal(ch1, 0)
# print(f"change_prob: \n {ch1}")
where_should_weights_change = np.where(ch1 == 1)
weights = wei1.copy()
weights[where_should_weights_change] = np.random.normal(wei1, 1, (neuron_count, neuron_count))[where_should_weights_change]
# weights = np.multiply(weights, ch1)
weights = np.multiply(weights, link1)
# print(f"weights: \n {wei_moves}")
ch1b = np.zeros((1, neuron_count))
ch1b[np.random.random((1, neuron_count)) <= 0.75] = 1
ch1b[0, :input_size] = 0
where_should_bias_change = np.where(ch1b == 1)
biases = bia1.copy()
biases[where_should_bias_change] = np.random.normal(bia1, 1, (1, neuron_count))[where_should_bias_change]
ch_it = random.random()
if ch_it <= 0.75:
    print(f"change_it: \n{try_choose_different(maxit, [1, 2, 3, 4, 5])}")
ch_l = np.zeros(wei1.shape)
ch_l[np.random.random(wei1.shape) <= 0.75] = 1
ch_l[:, :input_size] = 0
ch_l[-output_size:, :] = 0
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
new_edges_weights[added_edges] = np.random.uniform(minW, maxW, (neuron_count, neuron_count))[added_edges]
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
    if new_hc < hidden_size:
        to_preserve = [0, 1, 2, 3, 4]
        brr = choose_without_repetition(list(range(input_size, input_size + hidden_size)), hidden_size - new_hc)
        for i in brr:
            to_preserve.remove(i)
        to_preserve = np.array(to_preserve).reshape(1, -1)

        links = links[to_preserve[0, :, None], to_preserve]
        weights = weights[to_preserve[0, :, None], to_preserve]
        biases = biases[0, to_preserve]
        new_af = []
        print(f"to_preserve: {to_preserve}")

    elif new_hc > hidden_size:
        act_set = [ReLu(), Sigmoid(), GaussAct(), TanH()]
        link3 = get_links(input_size, output_size, input_size + output_size + new_hc)
        link3[:input_size + hidden_size, :input_size + hidden_size] = links[:input_size + hidden_size, :input_size + hidden_size]
        link3[:input_size + hidden_size, -output_size:] = links[:input_size + hidden_size, -output_size:]

        links = link3

        minW = np.min(weights)
        maxW = np.max(weights)
        weight3 = np.random.uniform(minW, maxW, (1 + 2 + new_hc, 1 + 2 + new_hc))
        weight3[:input_size + hidden_size, :input_size + hidden_size] = weights[:input_size + hidden_size, :input_size + hidden_size]
        weight3[:input_size + hidden_size, -output_size:] = weights[:input_size + hidden_size, -output_size:]
        weight3 = np.multiply(weight3, link3)

        weights = weight3

        minB = np.min(biases)
        maxB = np.max(biases)
        bia3 = np.random.uniform(minB, maxB, (1, input_size + output_size + new_hc))
        bia3[0, :input_size + hidden_size] = biases[0, :input_size + hidden_size]
        bia3[0, -output_size:] = biases[0, -output_size:]

        biases = bia3

        for i in range(new_hc - hidden_size):
            print(f"inc_add_{i+1}: {act_set[random.randint(0, len(act_set) - 1)]}")


print(f"new_links: \n {links}")
print(f"new_weights: \n{weights}")
print(f"biases: \n {biases}")
mut_rad_change = random.random()
if mut_rad_change <= 0.75:
    print(f"mut_rad: \n {random.uniform(0, 1)}")
wb_prob_change = random.random()
if wb_prob_change <= 0.75:
    print(f"wb_prob: \n {random.uniform(0.05, 0.1)}")
s_prob_change = random.random()
if s_prob_change <= 0.75:
    print(f"s_prob: \n {random.uniform(0.6, 0.7)}")
test_struct_mutation()
test_struct_mutation_2()
test_struct_mutation_3()


