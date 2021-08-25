import random
from math import log10

import numpy as np
# import pytest

# from ann_point.AnnPoint2 import AnnPoint2
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import *
from utility.Mut_Utility import *
from utility.TestingUtility import assert_chaos_network_properties

def test_struct_mutation():
    random.seed(20021234)
    np.random.seed(20021234)

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 sqr_mut_prob=(log10(0.4), log10(0.8)), lin_mut_prob=(log10(0.65), log10(0.95)),
                                 p_mutation_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 dstr_mut_prob=(log10(0.79), log10(0.81)))

    link1 = np.array([[0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 0],
                     [0, 0, 3, 0, 5],
                     [0, 7, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), net_it=2, mutation_radius=log10(0.5), sqr_mut_prob=log10(0.5),
                   lin_mut_prob=log10(0.8), p_mutation_prob=log10(0.7), c_prob=log10(0.3), dstr_mut_prob=log10(0.8))

    mo = FinalMutationOperator(hrange)
    mutant = mo.mutate(cn1)

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0, 1, 1, 0, 0],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0., 1, 2, 0, 0],
                                                    [0, 0, 3, 0, 5],
                                                    [0, 7, 0, 0, 6],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5]]),
                                    desired_actFun=[None, ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.5),
                                    desired_wb_prob=log10(0.5),
                                    desired_s_prob=log10(0.8),
                                    desired_p_prob=log10(0.7),
                                    desired_c_prob=log10(0.3),
                                    desired_r_prob=log10(0.8))


    # wb_pm = 0.5
    # s_pm = 0.8
    # p_pm = 0.7
    # r_pm = 0.4
    # c_pm = 0.3

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=5,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=3,
                                    desired_hidden_count=2,
                                    desired_links=np.array([[0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 1., 0.],
                                                  [0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0., 0., 0., 0.],
                                                    [0., 0., 0., 0., 0.],
                                                    [0., 0., 0., 6.88475867, 0.],
                                                    [0., 0., 0., 0., 0.],
                                                    [0., 0., 0., 0., 0.]]),
                                    desired_biases=np.array([[0, -1.1922916, -4.12076712, -4.55715203, -5.49155904]]),
                                    desired_actFun=[None, GaussAct(), GaussAct(), None, None],
                                    desired_aggr=GaussAct(),
                                    desired_maxit=5,
                                    desired_mut_rad=-0.331788201,
                                    desired_wb_prob=-0.2829968,
                                    desired_s_prob=-0.09691001,
                                    desired_p_prob=-0.152270485,
                                    desired_c_prob=-0.5772129,
                                    desired_r_prob=-0.09713193)

def test_struct_mutation_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 sqr_mut_prob=(log10(0.4), log10(0.8)), lin_mut_prob=(log10(0.65), log10(0.95)),
                                 p_mutation_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 dstr_mut_prob=(log10(0.59), log10(0.61)))

    link1 = np.array([[0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 2, 3, 4, 0, 0],
                     [0, 0, 5, 6, 0, 7],
                     [0, 8, 0, 0, 0, 9],
                     [0, 2, 0, 0, 3, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
                   aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), sqr_mut_prob=log10(0.75),
                   lin_mut_prob=log10(0.66), p_mutation_prob=log10(0.74),
                   c_prob=log10(0.15), dstr_mut_prob=log10(0.595))

    mo = FinalMutationOperator(hrange)

    random.seed(20021234)
    np.random.seed(20021234)
    mutant = mo.mutate(cn1)

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 0, 1],
                                                  [0, 1, 0, 0, 0, 1],
                                                  [0, 1, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 2, 3, 4, 0, 0],
                                                    [0, 0, 5, 6, 0, 7],
                                                    [0, 8, 0, 0, 0, 9],
                                                    [0, 2, 0, 0, 3, 5],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.35),
                                    desired_wb_prob=log10(0.75),
                                    desired_s_prob=log10(0.66),
                                    desired_p_prob=log10(0.74),
                                    desired_c_prob=log10(0.15),
                                    desired_r_prob=log10(0.595))

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0., 1., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 1., 0.],
                                                  [0., 1., 0., 1., 0., 1.],
                                                  [0., 0., 1., 0., 0., 1.],
                                                  [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[ 0.,          2.23563742,  0.        ,  0.        ,  0.        ,  0.        ],
                                                    [-0.,          0.        ,  0.        ,  0.        ,  1.72372606,  0.        ],
                                                    [-0.,          7.91732588,  0.        ,  7.49100203,  -0.        ,  9.55098225],
                                                    [-0.,          0.        ,  3.70193457,  0.        ,  0.        ,  4.65590867],
                                                    [ 0.,         -0.        , -0.        ,  0.        ,  0.        ,  0.        ],
                                                    [-0.,          0.        , -0.        ,  0.        ,  0.        ,  0.        ]]),
                                    desired_biases=np.array([[ 0.        , -1.70775246, -3.09250415, -4.        , -5.        , -6.        ]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=ReLu(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.45593195,
                                    desired_wb_prob=-0.1498469,
                                    desired_s_prob=-0.1822111,
                                    desired_p_prob=-0.132314,
                                    desired_c_prob=-0.881580,
                                    desired_r_prob=-0.2266123)

def test_struct_mutation_3():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 sqr_mut_prob=(log10(0.4), log10(0.8)), lin_mut_prob=(log10(0.65), log10(0.95)),
                                 p_mutation_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 dstr_mut_prob=(log10(0.59), log10(0.61)))

    link1 = np.array([[0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [0, 1, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 2, 3, 4, 0, 0],
                     [0, 0, 5, 6, 0, 7],
                     [0, 8, 0, 0, 0, 9],
                     [0, 2, 0, 0, 3, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
                   aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), sqr_mut_prob=log10(0.75),
                   lin_mut_prob=log10(0.66), p_mutation_prob=log10(0.74),
                   c_prob=log10(0.15), dstr_mut_prob=log10(0.595))

    mo = FinalMutationOperator(hrange)

    random.seed(1112111)
    np.random.seed(1112111)
    mutant = mo.mutate(cn1)

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 0, 1],
                                                  [0, 1, 0, 0, 0, 1],
                                                  [0, 1, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 2, 3, 4, 0, 0],
                                                    [0, 0, 5, 6, 0, 7],
                                                    [0, 8, 0, 0, 0, 9],
                                                    [0, 2, 0, 0, 3, 5],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.35),
                                    desired_wb_prob=log10(0.75),
                                    desired_s_prob=log10(0.66),
                                    desired_p_prob=log10(0.74),
                                    desired_c_prob=log10(0.15),
                                    desired_r_prob=log10(0.595))

    assert_chaos_network_properties(net=mutant,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0., 1., 0., 0., 0., 0.],
                                                  [0., 0., 1., 1., 0., 1.],
                                                  [0., 0., 0., 1., 1., 0.],
                                                  [0., 1., 1., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[ 0.         , 1.86811848,  0.        ,  0.        , -0.        ,  0.        ],
                                                     [ 0.         , 0.        ,  5.07223363,  6.09991906, -0.        ,  6.69803919],
                                                     [-0.         , 0.        , -0.        ,  4.16703127,  6.19592067 ,  0.        ],
                                                     [-0.         , 2.        ,  8.82747901, -0.        ,  0.        ,  0.        ],
                                                     [-0.         ,-0.        , -0.        ,  0.        , -0.        , -0.        ],
                                                     [-0.         , 0.        ,  0.        ,  0.        , -0.        ,  0.        ]]),
                                    desired_biases=np.array([[-0.        , -2.63499526, -3.        , -3.79587563, -5.10905153, -6.25300831]]),
                                    desired_actFun=[None, TanH(), Sigmoid(), GaussAct(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.44981784,
                                    desired_wb_prob=-0.1300882,
                                    desired_s_prob=-0.182737,
                                    desired_p_prob=-0.1306823,
                                    desired_c_prob=-0.8446919,
                                    desired_r_prob=-0.2245557)




seed = 20021234
random.seed(seed)
np.random.seed(seed)



hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                             sqr_mut_prob=(log10(0.4), log10(0.8)), lin_mut_prob=(log10(0.65), log10(0.95)),
                             p_mutation_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                             dstr_mut_prob=(log10(0.59), log10(0.61)))

link1 = np.array([[0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
wei1 = np.array([[0, 2, 3, 4, 0, 0],
                 [0, 0, 5, 6, 0, 7],
                 [0, 8, 0, 0, 0, 9],
                 [0, 2, 0, 0, 3, 5],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0.]])
bia1 = np.array([[0., -2, -3, -4, -5, -6]])
actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
               aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), sqr_mut_prob=log10(0.75),
               lin_mut_prob=log10(0.66), p_mutation_prob=log10(0.74),
               c_prob=log10(0.15), dstr_mut_prob=log10(0.595))

aggr1 = cn1.aggrFun
net_it = cn1.net_it
input_size = cn1.input_size
hidden_size = cn1.hidden_count
output_size = cn1.output_size
neuron_count = cn1.neuron_count
frac=0.1

radius = 10 ** cn1.mutation_radius
sqr_mut_prob = 10 ** cn1.sqr_mut_prob
lin_mut_prob = 10 ** cn1.lin_mut_prob
p_mutation_prob = 10 ** cn1.p_mutation_prob
c_prob = 10 ** cn1.c_prob
dstr_mut_prob = 10 ** cn1.dstr_mut_prob


ws = gaussian_shift(wei1, link1, sqr_mut_prob, radius)
bf = gaussian_shift(bia1, get_bias_mask(input_size, neuron_count), lin_mut_prob, radius)

naf = input_size * [None]
for i in range(input_size, input_size + hidden_size):
    naf.append(conditional_try_choose_different(dstr_mut_prob, actFuns1[i], hrange.actFunSet))
naf.extend(output_size * [None])

nag = conditional_try_choose_different(dstr_mut_prob, aggr1, hrange.actFunSet)

lf, wf = add_or_remove_edges(dstr_mut_prob, link1, ws, get_weight_mask(input_size, output_size, neuron_count), hrange)

nnit = conditional_try_choose_different(lin_mut_prob, net_it, list(range(hrange.min_it, hrange.max_it + 1)))

nmr = conditional_uniform_value_shift(p_mutation_prob, log10(radius), hrange.min_mut_radius, hrange.max_mut_radius, frac)

nsqr = conditional_uniform_value_shift(p_mutation_prob, log10(sqr_mut_prob), hrange.min_sqr_mut_prob, hrange.max_sqr_mut_prob, frac)

nlin = conditional_uniform_value_shift(p_mutation_prob, log10(lin_mut_prob), hrange.min_lin_mut_prob, hrange.max_lin_mut_prob, frac)

npp = conditional_uniform_value_shift(p_mutation_prob, log10(p_mutation_prob), hrange.min_p_mut_prob, hrange.max_p_mut_prob, frac)

ncp = conditional_uniform_value_shift(p_mutation_prob, log10(c_prob), hrange.min_c_prob, hrange.max_c_prob, frac)

ndstr = conditional_uniform_value_shift(p_mutation_prob, log10(dstr_mut_prob), hrange.min_dstr_mut_prob, hrange.max_dstr_mut_prob, frac)

print("lf")
print(lf)
print("wf")
print(wf)
print("bf")
print(bf)
print("naf")
print(naf)
print("nag")
print(nag)
print("nnit")
print(nnit)
print("nmr")
print(nmr)
print("nsqr")
print(nsqr)
print("nlin")
print(nlin)
print("npp")
print(npp)
print("ncp")
print(ncp)
print("ndstr")
print(ndstr)

# test_struct_mutation_2()



