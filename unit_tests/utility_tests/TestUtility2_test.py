import random

from neural_network.ChaosNet import *
# from utility.Mut_Utility import inflate_network
from utility.TestingUtility import compare_chaos_network, compare_chaos_networks
import numpy as np

def test_chaos_compare_1():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    # try:
    #     compare_chaos_networks(net=cn1, net2=cn2)
    # except AssertionError:
    #     assert True
    # else:
    #     assert False



    compare_chaos_networks(net=cn1, net2=cn2)

def test_chaos_compare_2():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=1, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_3():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)


    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=2, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_4():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    cn2.neuron_count = 1

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_5():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.hidden_start_index = 0

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_6():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.hidden_end_index = -3

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_7():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.hidden_count = -14

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_8():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 1, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

# def test_chaos_compare_9():
#     link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     link2 = np.array([[0, 0, 1],
#                       [0, 1, 0],
#                       [0, 0, 1],
#                       [0, 1, 0],
#                       [0, 0 , 0 ]])
#     wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
#                    actFuns=actFuns2, aggrFun=Sigmoid().copy(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     try:
#         compare_chaos_networks(net=cn1, net2=cn2)
#     except AssertionError:
#         assert True
#     else:
#         assert False

def test_chaos_compare_10():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, -1, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

# def test_chaos_compare_11():
#     link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
#                    actFuns=actFuns2, aggrFun=Sigmoid().copy(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     try:
#         compare_chaos_networks(net=cn1, net2=cn2)
#     except AssertionError:
#         assert True
#     else:
#         assert False

def test_chaos_compare_12():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 1, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

# def test_chaos_compare_13():
#     link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
#                    actFuns=actFuns2, aggrFun=Sigmoid().copy(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     try:
#         compare_chaos_networks(net=cn1, net2=cn2)
#     except AssertionError:
#         assert True
#     else:
#         assert False

# def test_chaos_compare_14():
#     link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
#                    actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
#                       [0, 0, 0, 1, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 1],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
#                       [0, 0, 0, 1, 0, 9 , 0 ],
#                       [0, 0, 0, 2, 6, 0 , 13],
#                       [0, 0, 0, 0, 7, 10, 0 ],
#                       [0, 0, 0, 3, 0, 11, 14],
#                       [0, 0, 0, 4, 8, 0 , 15],
#                       [0, 0, 0, 0, 0, 0 , 0 ]])
#     bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
#     actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]
#
#     cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
#                    actFuns=actFuns2, aggrFun=Sigmoid().copy(), maxit=10,
#                    mutation_radius=-1, wb_mutation_prob=-2.5, s_mutation_prob=-1, p_mutation_prob=-0.44,
#                    c_prob=-11, r_prob=-22)
#
#     try:
#         compare_chaos_networks(net=cn1, net2=cn2)
#     except AssertionError:
#         assert True
#     else:
#         assert False

def test_chaos_compare_15():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), Poly2(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_16():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=LReLu().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_17():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=5,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_18():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-11, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_19():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.53, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_20():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-16, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_21():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    cn2.inp[0, -1] = 1

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_22():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.inp = np.zeros((1, 3))

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except ValueError:
        assert True
    else:
        assert False

def test_chaos_compare_23():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.act[0, -4] = 222

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_24():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)
    cn2.act = np.zeros((3, 3))
    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except ValueError:
        assert True
    else:
        assert False

def test_chaos_compare_25():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.4442,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False



def test_chaos_compare_26():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-111, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False



def test_chaos_compare_27():
    link1 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei1 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia1 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns1 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn1 = ChaosNet(input_size=3, output_size=1, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=Sigmoid(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-22)

    link2 = np.array([[0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 0, 1, 0],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    wei2 =  np.array([[0, 0, 0, 0, 5, 0 , 12],
                      [0, 0, 0, 1, 0, 9 , 0 ],
                      [0, 0, 0, 2, 6, 0 , 13],
                      [0, 0, 0, 0, 7, 10, 0 ],
                      [0, 0, 0, 3, 0, 11, 14],
                      [0, 0, 0, 4, 8, 0 , 15],
                      [0, 0, 0, 0, 0, 0 , 0 ]])
    bia2 = np.array([[0., 0, 0, -1, -2, -3, -4]])
    actFuns2 = [None, None,  None, ReLu(), SincAct(), ReLu(), None]

    cn2 = ChaosNet(input_size=3, output_size=1, links=link2.copy(), weights=wei2.copy(), biases=bia2.copy(),
                   actFuns=actFuns2, aggrFun=Sigmoid().copy(), net_it=10,
                   mutation_radius=-1, sqr_mut_prob=-2.5, lin_mut_prob=-1, p_mutation_prob=-0.44,
                   c_prob=-11, dstr_mut_prob=-222)

    try:
        compare_chaos_networks(net=cn1, net2=cn2)
    except AssertionError:
        assert True
    else:
        assert False


test_chaos_compare_1()
test_chaos_compare_2()
test_chaos_compare_3()
test_chaos_compare_4()
test_chaos_compare_5()
test_chaos_compare_6()
test_chaos_compare_7()
test_chaos_compare_8()
# test_chaos_compare_9()
test_chaos_compare_10()
# test_chaos_compare_11()
test_chaos_compare_12()
# test_chaos_compare_13()
# test_chaos_compare_14()
test_chaos_compare_15()
test_chaos_compare_16()
test_chaos_compare_17()
test_chaos_compare_18()
test_chaos_compare_19()
test_chaos_compare_20()
test_chaos_compare_21()
test_chaos_compare_22()
test_chaos_compare_23()
test_chaos_compare_24()
test_chaos_compare_25()
test_chaos_compare_26()
test_chaos_compare_27()