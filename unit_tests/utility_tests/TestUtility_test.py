import random

from neural_network.ChaosNet import *
from utility.Mut_Utility import inflate_network
from utility.TestingUtility import compare_chaos_network
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=3,
                          desited_output_size=1,
                          desired_neuron_count=7,
                          desired_hidden_start_index=3,
                          desired_hidden_end_index=6,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                  [0, 0, 0, 1, 0, 1, 0],
                                                  [0, 0, 0, 1, 1, 0, 1],
                                                  [0, 0, 0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 0, 1, 1],
                                                  [0, 0, 0, 1, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0 , 0 ]]),
                          desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                    [0, 0, 0, 1, 0, 9 , 0 ],
                                                    [0, 0, 0, 2, 6, 0 , 13],
                                                    [0, 0, 0, 0, 7, 10, 0 ],
                                                    [0, 0, 0, 3, 0, 11, 14],
                                                    [0, 0, 0, 4, 8, 0 , 15],
                                                    [0, 0, 0, 0, 0, 0 , 0 ]]),
                          desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                          desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                          desired_aggr=Sigmoid(),
                          desired_maxit=10,
                          desired_mut_rad=-1,
                          desired_wb_prob=2.5,
                          desired_s_prob=1,
                          desired_p_prob=0.44,
                          desired_c_prob=-11,
                          desired_r_prob=22,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))

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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=0,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=2,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=-11,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=6,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=-6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=-3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 1, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_9():
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 1],
                                                      [0, 1, 0],
                                                      [0, 0, 1],
                                                      [0, 1, 0],
                                                      [0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False

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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, -1, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_11():
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 12],
                                                        [0, 0, 0 ],
                                                        [0, 0, 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except ValueError:
        assert True
    else:
        assert False

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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 1, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False

def test_chaos_compare_13():
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except ValueError:
        assert True
    else:
        assert False

def test_chaos_compare_14():
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False

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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), Sigmoid(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=LReLu(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=-10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-10,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.55,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=111,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 1.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0., 0]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., -222., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.444,
                              desired_c_prob=-11,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-111,
                              desired_r_prob=22,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
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
                   actFuns=actFuns1, aggrFun=Sigmoid(), maxit=10,
                   mutation_radius=-1, wb_mutation_prob=2.5, s_mutation_prob=1, p_mutation_prob=0.44,
                   c_prob=-11, r_prob=22)

    np.random.seed(1001)
    random.seed(1001)

    ##########################################################################

    try:
        compare_chaos_network(net=cn1,
                              desired_input_size=3,
                              desited_output_size=1,
                              desired_neuron_count=7,
                              desired_hidden_start_index=3,
                              desired_hidden_end_index=6,
                              desired_hidden_count=3,
                              desired_links=np.array([[0, 0, 0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0, 1, 0],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 1, 1, 0],
                                                      [0, 0, 0, 1, 0, 1, 1],
                                                      [0, 0, 0, 1, 1, 0, 1],
                                                      [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_weights=np.array([[0, 0, 0, 0, 5, 0 , 12],
                                                        [0, 0, 0, 1, 0, 9 , 0 ],
                                                        [0, 0, 0, 2, 6, 0 , 13],
                                                        [0, 0, 0, 0, 7, 10, 0 ],
                                                        [0, 0, 0, 3, 0, 11, 14],
                                                        [0, 0, 0, 4, 8, 0 , 15],
                                                        [0, 0, 0, 0, 0, 0 , 0 ]]),
                              desired_biases=np.array([[0., 0, 0, -1, -2, -3, -4]]),
                              desired_actFun=[None, None,  None, ReLu(), SincAct(), ReLu(), None],
                              desired_aggr=Sigmoid(),
                              desired_maxit=10,
                              desired_mut_rad=-1,
                              desired_wb_prob=2.5,
                              desired_s_prob=1,
                              desired_p_prob=0.44,
                              desired_c_prob=-11,
                              desired_r_prob=222,
                              desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                              desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    except AssertionError:
        assert True
    else:
        assert False


#
# test_chaos_compare_2()
# test_chaos_compare_3()
# test_chaos_compare_4()
# test_chaos_compare_5()
# test_chaos_compare_6()
# test_chaos_compare_7()
# test_chaos_compare_8()
# test_chaos_compare_9()
# test_chaos_compare_10()
# test_chaos_compare_11()
# test_chaos_compare_12()
# test_chaos_compare_13()
# test_chaos_compare_14()
# test_chaos_compare_15()
# test_chaos_compare_16()
# test_chaos_compare_17()
# test_chaos_compare_18()
# test_chaos_compare_19()
# test_chaos_compare_20()
# test_chaos_compare_21()
# test_chaos_compare_22()
# test_chaos_compare_23()
# test_chaos_compare_24()
# test_chaos_compare_25()
# test_chaos_compare_26()
# test_chaos_compare_27()