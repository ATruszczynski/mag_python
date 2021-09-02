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
    seed = 20021
    random.seed(seed)
    np.random.seed(seed)

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(1), log10(2)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.79), log10(0.81)))

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
                   actFuns=actFuns1, aggrFun=TanH(), net_it=2, mutation_radius=log10(0.5), swap_prob=log10(0.5),
                   multi=log10(1.5), p_prob=log10(0.7), c_prob=log10(0.3), p_rad=log10(0.8))

    mo = FinalMutationOperator(hrange)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

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
                                    desired_s_prob=log10(1.5),
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
                                    desired_links=np.array([[0., 0., 1., 0., 0.],
                                                            [0., 0., 1., 0., 1.],
                                                            [0., 1., 0., 0., 1.],
                                                            [0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0, 2, 0, 0],
                                                              [0, 0, 3, 0, 5],
                                                              [0, 6.97400427, 0, 0, 6],
                                                              [0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0]]),
                                    desired_biases=np.array([[0, -2, -3.19269902, -4, -4.59589825]]),
                                    desired_actFun=[None, ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=-0.25782364,
                                    desired_wb_prob=-0.22687510,
                                    desired_s_prob=0.176091259,
                                    desired_p_prob=-0.17788975,
                                    desired_c_prob=-0.42450710,
                                    desired_r_prob=-0.09559582)

def test_struct_mutation_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(0.65), log10(100)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.59), log10(0.61)))

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
                   aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), swap_prob=log10(0.75),
                   multi=log10(90), p_prob=log10(0.74),
                   c_prob=log10(0.15), p_rad=log10(0.595))

    mo = FinalMutationOperator(hrange)


    random.seed(20021234)
    np.random.seed(20021234)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

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
                                    desired_s_prob=log10(90),
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
                                    desired_links=np.array([[0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 1., 0.],
                                                            [0., 0., 0., 1., 1., 0.],
                                                            [0., 0., 1., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 1.72372606, 0.],
                                                              [0., 0., 0., 7.49100203, 5.29989391, 0.],
                                                              [0., 0., 3.70193457, 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.]]),
                                    desired_biases=np.array([[-0.        , -1.70775246, -3.09250415, -3.53797184 ,-5.79764051 ,-5.8394995 ]]),
                                    desired_actFun=[None, GaussAct(), TanH(), GaussAct(), None, None],
                                    desired_aggr=Sigmoid(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.4956634,
                                    desired_wb_prob=-0.22375686,
                                    desired_s_prob=0.6812745485,
                                    desired_p_prob=-0.16303863,
                                    desired_c_prob=-0.76943127,
                                    desired_r_prob=-0.222210888)

def test_struct_mutation_3():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                                 swap=(log10(0.4), log10(0.8)), multi=(log10(0.65), log10(1000)),
                                 p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                                 p_rad=(log10(0.59), log10(0.61)))

    link1 = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [0, 1, 1, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 2, 0, 0, 0, 0],
                     [0, 0, 5, 6, 0, 7],
                     [0, 8, 0, 0, 0, 9],
                     [0, 2, -1, 0, 3, 5],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
                   aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), swap_prob=log10(0.75),
                   multi=log10(99), p_prob=log10(0.74),
                   c_prob=log10(0.15), p_rad=log10(0.595))

    mo = FinalMutationOperator(hrange)

    random.seed(1112111)
    np.random.seed(1112111)
    mutant = mo.mutate(cn1)

    cn1 = cn1.copy()

    assert_chaos_network_properties(net=cn1,
                                    desired_input_size=1,
                                    desired_output_size=2,
                                    desired_neuron_count=6,
                                    desired_hidden_start_index=1,
                                    desired_hidden_end_index=4,
                                    desired_hidden_count=3,
                                    desired_links=np.array([[0, 1, 0, 0, 0, 0],
                                                            [0, 0, 1, 1, 0, 1],
                                                            [0, 1, 0, 0, 0, 1],
                                                            [0, 1, 1, 0, 1, 1],
                                                            [0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0]]),
                                    desired_weights=np.array([[0, 2, 0, 0, 0, 0],
                                                              [0, 0, 5, 6, 0, 7],
                                                              [0, 8, 0, 0, 0, 9],
                                                              [0, 2, -1, 0, 3, 5],
                                                              [0, 0, 0, 0, 0, 0],
                                                              [0, 0, 0, 0, 0, 0.]]),
                                    desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                                    desired_actFun=[None, ReLu(), ReLu(), TanH(), None, None],
                                    desired_aggr=TanH(),
                                    desired_maxit=2,
                                    desired_mut_rad=log10(0.35),
                                    desired_wb_prob=log10(0.75),
                                    desired_s_prob=log10(99),
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
                                    desired_links=np.array([[0., 0., 1., 1., 0., 0.],
                                                            [0., 0., 0., 0., 1., 0.],
                                                            [0., 0., 0., 1., 1., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.],
                                                            [0., 0., 0., 0., 0., 0.]]),
                                    desired_weights=np.array([[0., 0., 5.03743713, -0.01032964, 0., 0.],
                                                              [0., 0., 0., 0., 4.4777615, 0.],
                                                              [0., 0., 0., 2.52434007, 5.43532301, 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.],
                                                              [0., 0., 0., 0., 0., 0.]]),
                                    desired_biases=np.array([[-0.        , -2.63499526, -3.3678456 , -3.79587563, -5.10905153, -6.25300831]]),
                                    desired_actFun=[None, TanH(), Sigmoid(), GaussAct(), None, None],
                                    desired_aggr=Sigmoid(),
                                    desired_maxit=1,
                                    desired_mut_rad=-0.522410350,
                                    desired_wb_prob=-0.10412866,
                                    desired_s_prob=1.14319451,
                                    desired_p_prob=-0.1458257,
                                    desired_c_prob=-0.62158282,
                                    desired_r_prob=-0.22909079)




seed = 1112111
random.seed(seed)
np.random.seed(seed)




hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 5), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(log10(0.25), log10(1)),
                             swap=(log10(0.4), log10(0.8)), multi=(log10(0.65), log10(1000)),
                             p_prob=(log10(0.65), log10(0.75)), c_prob=(log10(0.1), log10(0.4)),
                             p_rad=(log10(0.59), log10(0.61)))

link1 = np.array([[0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 1],
                  [0, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
wei1 = np.array([[0, 2, 0, 0, 0, 0],
                 [0, 0, 5, 6, 0, 7],
                 [0, 8, 0, 0, 0, 9],
                 [0, 2, -1, 0, 3, 5],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0.]])
bia1 = np.array([[0., -2, -3, -4, -5, -6]])
actFuns1 = [None, ReLu(), ReLu(), TanH(), None, None]

cn1 = ChaosNet(input_size=1, output_size=2, links=link1, weights=wei1, biases=bia1, actFuns=actFuns1,
               aggrFun=TanH(), net_it=2, mutation_radius=log10(0.35), swap_prob=log10(0.75),
               multi=log10(99), p_prob=log10(0.74),
               c_prob=log10(0.15), p_rad=log10(0.595))
aggr1 = cn1.aggrFun
net_it = cn1.net_it
input_size = cn1.input_size
hidden_size = cn1.hidden_count
output_size = cn1.output_size
neuron_count = cn1.neuron_count

depr =    10 ** cn1.swap_prob
modi_nc =    10 ** cn1.multi
p_mutation_prob =      10 ** cn1.p_prob
p_rad =  10 ** cn1.p_rad
mutation_radius =    10 ** cn1.mutation_radius
c_prob = 10 ** cn1.c_prob

func = cn1.neuron_count - cn1.input_size

sqr_mut_prob = modi_nc / func
dstr_mut_prob = modi_nc / func ** 2

ws = gaussian_shift(wei1, link1, sqr_mut_prob, mutation_radius)
bf = gaussian_shift(bia1, get_bias_mask(input_size, neuron_count), sqr_mut_prob, mutation_radius)

naf = input_size * [None]
for i in range(input_size, input_size + hidden_size):
    naf.append(conditional_try_choose_different(dstr_mut_prob, actFuns1[i], hrange.actFunSet))
naf.extend(output_size * [None])

nag = conditional_try_choose_different(dstr_mut_prob, aggr1, hrange.actFunSet)

lf, wf = add_or_remove_edges(dstr_mut_prob, link1, ws, get_weight_mask(input_size, output_size, neuron_count), hrange)

minn = max(hrange.min_it, net_it - 1)
maxn = min(hrange.max_it, net_it + 1)
nnit = conditional_try_choose_different(dstr_mut_prob, net_it, list(range(minn, maxn + 1)))


nmr = conditional_uniform_value_shift(p_mutation_prob, log10(mutation_radius), hrange.min_mut_radius, hrange.max_mut_radius, p_rad)

nsqr = conditional_uniform_value_shift(p_mutation_prob, log10(depr), hrange.min_swap, hrange.max_swap, p_rad)

nlin = conditional_uniform_value_shift(p_mutation_prob, log10(modi_nc), hrange.min_multi, hrange.max_multi, p_rad)

npp = conditional_uniform_value_shift(p_mutation_prob, log10(p_mutation_prob), hrange.min_p_prob, hrange.max_p_prob, p_rad)

ncp = conditional_uniform_value_shift(p_mutation_prob, log10(c_prob), hrange.min_c_prob, hrange.max_c_prob, p_rad)

ndstr = conditional_uniform_value_shift(p_mutation_prob, log10(p_rad), hrange.min_p_rad, hrange.max_p_rad, p_rad)

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

# test_struct_mutation()
# test_struct_mutation_2()
# test_struct_mutation_3()



