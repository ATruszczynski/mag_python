import pytest

from evolving_classifier.operators.CrossoverOperator import find_possible_cuts
from utility.Mut_Utility import *
from utility.TestingUtility import compare_chaos_network
from utility.Utility import *


def test_neuron_increase():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))

    link1 = np.array([[0, 1, 1, 0, 1],
                      [0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 4],
                     [0, 0, 3, 8, 5],
                     [0, 7, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), Sigmoid(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    np.random.seed(1001)
    random.seed(1001)


    cn2 = change_neuron_count(cn1, hrange, 4)


    ############################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 1, 0, 1],
                                                  [0, 0, 1, 1, 1],
                                                  [0, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0., 1, 2, 0, 4],
                                                    [0, 0, 3, 8, 5],
                                                    [0, 7, 0, 0, 6],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -4, -5]]),
                          desired_actFun=[None, ReLu(), Sigmoid(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6)


    ############################################################################

    # assert cn2.input_size == 1
    # assert cn2.output_size == 2
    # assert cn2.neuron_count == 7
    # assert cn2.hidden_start_index == 1
    # assert cn2.hidden_end_index == 5
    # assert cn2.hidden_count == 4
    #
    # assert cn2.maxit == 2
    # assert cn2.hidden_comp_order is None
    #
    # assert np.array_equal(cn2.links, np.array([[0., 1., 1., 1., 1., 0., 1.],
    #                                            [0., 0., 1., 0., 1., 1., 1.],
    #                                            [0., 1., 0., 1., 0., 0., 1.],
    #                                            [0., 1., 1., 0., 0., 1., 1.],
    #                                            [0., 0., 1., 1., 0., 1., 1.],
    #                                            [0., 0., 0., 0., 0., 0., 0.],
    #                                            [0., 0., 0., 0., 0., 0., 0.]]))
    #
    # assert np.all(np.isclose(cn2.weights, np.array([[0., 1, 2, 1.99852202, 3.53627653, 0, 4],
    #                                                 [0 , 0, 3, 0, 2.08113262, 8, 5],
    #                                                 [0 , 7, 0, 0.87783263, 0, 0, 6],
    #                                                 [0., 7.15815828, 0.34053276, 0., 0., 5.35134103, 6.56899812],
    #                                                 [0., 0., 5.58797454, 4.99502278, 0., 7.54306312, 6.34765188],
    #                                                 [0 , 0, 0, 0, 0, 0, 0],
    #                                                 [0 , 0, 0, 0, 0, 0, 0]])))
    #
    # assert np.all(np.isclose(cn2.biases, np.array([[0., -2, -3, -0.54751109, -1.92308629, -4, -5]])))
    #
    # assert np.array_equal(cn2.inp, np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    # assert np.array_equal(cn2.act, np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    #
    # assert len(cn2.actFuns) == 7
    # assert cn2.actFuns[0] is None
    # assert cn2.actFuns[1].to_string() == ReLu().to_string()
    # assert cn2.actFuns[2].to_string() == Sigmoid().to_string()
    # assert cn2.actFuns[3].to_string() == SincAct().to_string()
    # assert cn2.actFuns[4].to_string() == ReLu().to_string()
    # assert cn2.actFuns[5] is None
    # assert cn2.actFuns[6] is None
    #
    # assert cn2.aggrFun.to_string() == TanH().to_string()


    compare_chaos_network(net=cn2,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=5,
                          desired_hidden_count=4,
                          desired_links=np.array([[0., 1., 1., 1., 1., 0., 1.],
                                                  [0., 0., 1., 0., 1., 1., 1.],
                                                  [0., 1., 0., 1., 0., 0., 1.],
                                                  [0., 1., 1., 0., 0., 1., 1.],
                                                  [0., 0., 1., 1., 0., 1., 1.],
                                                  [0., 0., 0., 0., 0., 0., 0.],
                                                  [0., 0., 0., 0., 0., 0., 0.]]),
                          desired_weights=np.array([[0., 1, 2, 1.99852202, 3.53627653, 0, 4],
                                                    [0 , 0, 3, 0, 2.08113262, 8, 5],
                                                    [0 , 7, 0, 0.87783263, 0, 0, 6],
                                                    [0., 7.15815828, 0.34053276, 0., 0., 5.35134103, 6.56899812],
                                                    [0., 0., 5.58797454, 4.99502278, 0., 7.54306312, 6.34765188],
                                                    [0 , 0, 0, 0, 0, 0, 0],
                                                    [0 , 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -0.54751109, -1.92308629, -4, -5]]),
                          desired_actFun=[None, ReLu(), Sigmoid(), SincAct(), ReLu(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))

def test_neuron_decrease():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))

    link1 = np.array([[0, 1, 1, 0, 0, 1],
                      [0, 0, 1, 1, 1, 1],
                      [0, 1, 0, 1, 0, 1],
                      [0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 =  np.array([[0, 1, 4, 0, 0, 10],
                      [0, 0, 5, 7, 9, 11],
                      [0, 2, 0, 8, 0, 12],
                      [0, 3, 6, 0, 0, 13],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0.]])
    bia1 = np.array([[0., -2, -3, -4, -5, -6]])
    actFuns1 = [None, ReLu(), Sigmoid(), SincAct(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = change_neuron_count(cn1, hrange, 1)

    ##########################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=6,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=4,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 1, 1, 0, 0, 1],
                                                  [0, 0, 1, 1, 1, 1],
                                                  [0, 1, 0, 1, 0, 1],
                                                  [0, 1, 1, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 1, 4, 0, 0, 10],
                                                    [0, 0, 5, 7, 9, 11],
                                                    [0, 2, 0, 8, 0, 12],
                                                    [0, 3, 6, 0, 0, 13],
                                                    [0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0.]]),
                          desired_biases=np.array([[0., -2, -3, -4, -5, -6]]),
                          desired_actFun=[None, ReLu(), Sigmoid(), SincAct(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0.]]))

    ##########################################################################

    compare_chaos_network(net=cn2,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=4,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=2,
                          desired_hidden_count=1,
                          desired_links=np.array([[0, 0, 0, 1],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0, 10],
                                                    [0, 0, 0, 13],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0.]]),
                          desired_biases=np.array([[0., -4, -5, -6]]),
                          desired_actFun=[None, SincAct(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0.]]))

    ##########################################################################

def test_network_inflate():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))

    link1 = np.array([[0, 1, 0, 0, 1],
                      [0, 0, 1, 1, 1],
                      [0, 1, 1, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 =  np.array([[0, 1, 0, 0, 6],
                      [0, 0, 3, 5, 7],
                      [0, 2, 4, 0, 8],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -4, -5, -6]])
    actFuns1 = [None, ReLu(), SincAct(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = inflate_network(cn1, 2)

    ##########################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 0, 0, 1],
                                                  [0, 0, 1, 1, 1],
                                                  [0, 1, 1, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 1, 0, 0, 6],
                                                    [0, 0, 3, 5, 7],
                                                    [0, 2, 4, 0, 8],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -4, -5, -6]]),
                          desired_actFun=[None, ReLu(), SincAct(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0.]]))


    ##########################################################################

    compare_chaos_network(net=cn2,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=5,
                          desired_hidden_count=4,
                          desired_links=np.array([[0, 1, 0, 0, 0, 0, 1],
                                                  [0, 0, 1, 0, 0, 1, 1],
                                                  [0, 1, 1, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 1, 0, 0, 0, 0, 6],
                                                    [0, 0, 3, 0, 0, 5, 7],
                                                    [0, 2, 4, 0, 0, 0, 8],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -4, 0, 0, -5, -6]]),
                          desired_actFun=[None, ReLu(), SincAct(), ReLu(), SincAct(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))

def test_network_deflate():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))

    link1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    wei1 =  np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 3, 0, 6, 0, 0, 0, 8],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 4, 0, 7, 0, 0, 0, 9],
                      [0, 0, 5, 0, 0, 0, 0, 0, 10],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    bia1 = np.array([[0., 0, -4, -5, -6, -7, -8, -9, -10]])
    actFuns1 = [None, None, SincAct(), TanH(), LReLu(), GaussAct(), Sigmoid(), None, None]

    cn1 = ChaosNet(input_size=2, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)




    np.random.seed(1001)
    random.seed(1001)

    cn2 = deflate_network(cn1)


    ##########################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=2,
                          desited_output_size=2,
                          desired_neuron_count=9,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=7,
                          desired_hidden_count=5,
                          desired_links=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 1, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 1, 0, 0, 0, 1],
                                                  [0, 0, 1, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 3, 0, 6, 0, 0, 0, 8],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 4, 0, 7, 0, 0, 0, 9],
                                                    [0, 0, 5, 0, 0, 0, 0, 0, 10],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., 0, -4, -5, -6, -7, -8, -9, -10]]),
                          desired_actFun=[None, None, SincAct(), TanH(), LReLu(), GaussAct(), Sigmoid(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]]))

    ##########################################################################

    compare_chaos_network(net=cn2,
                          desired_input_size=2,
                          desited_output_size=2,
                          desired_neuron_count=7,
                          desired_hidden_start_index=2,
                          desired_hidden_end_index=5,
                          desired_hidden_count=3,
                          desired_links=np.array([[0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 1, 0, 1],
                                                  [0, 0, 1, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 3, 0, 6, 0, 8],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 4, 0, 7, 0, 9],
                                                    [0, 0, 5, 0, 0, 0, 10],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., 0, -4, -5, -6, -9, -10]]),
                          desired_actFun=[None, None, SincAct(), TanH(), LReLu(), None, None],
                          desired_aggr=TanH(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6,
                          desired_inp=np.array([[0., 0., 0., 0., 0., 0., 0.]]),
                          desired_act=np.array([[0., 0., 0., 0., 0., 0., 0.]]))

def test_possible_cuts_1():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))
    link1 = np.array([[0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    bia1 = np.array([[-1., -2, -4, -5, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None, None]

    link2 = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0]])
    bia2 = np.array([[-10, -20, -30, -40]])
    actFuns2 = [None, None, None, None]

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 4
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 2, 0, 0])
    assert compare_lists(possible_cuts[1], [0, 3, 2, 1, 1, 0, 0])
    assert compare_lists(possible_cuts[2], [0, 4, 2, 2, 0, 0, 0])
    assert compare_lists(possible_cuts[3], [0, 5, 3, 2, 0, 0, 0])

def test_possible_cuts_1_2():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))

    link1 = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 0]])
    wei1 = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0]])
    bia1 = np.array([[-10, -20, -30, -40]])
    actFuns1 = [None, None, None, None]


    link2 = np.array([[0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
    bia2 = np.array([[-1., -2, -4, -5, -4, -5]])
    actFuns2 = [None, ReLu(), ReLu(), None, None, None]

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 4
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 0, 0, 2])
    assert compare_lists(possible_cuts[1], [0, 2, 3, 0, 0, 1, 1])
    assert compare_lists(possible_cuts[2], [0, 2, 4, 0, 0, 2, 0])
    assert compare_lists(possible_cuts[3], [0, 3, 5, 0, 0, 2, 0])


def test_possible_cuts_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))
    link1 = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 0, 4],
                     [0 , 0, 0, 5],
                     [0 , 0, 0, 0],
                     [0 , 0, 0, 0]])
    bia1 = np.array([[-1., -2, -4, -5]])
    actFuns1 = [None, ReLu(), None, None]

    link2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0,  0,  0 ],
                     [0, 0, 10, 20, 0 ],
                     [0, 0, 0,  30, 40],
                     [0, 0, 0,  0,  0 ],
                     [0, 0, 0,  0,  0.]])
    bia2 = np.array([[-10, -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 5
    assert compare_lists(possible_cuts[0], [0, 1, 1, 0, 1, 0, 2])
    assert compare_lists(possible_cuts[1], [0, 1, 2, 0, 1, 1, 1])
    assert compare_lists(possible_cuts[2], [0, 2, 2, 1, 0, 1, 1])
    assert compare_lists(possible_cuts[3], [0, 2, 3, 1, 0, 2, 0])
    assert compare_lists(possible_cuts[4], [0, 3, 4, 1, 0, 2, 0])



def test_possible_cuts_3():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()],mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))
    link1 = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    wei1 = np.array([[0, 1, 0, 1],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])
    bia1 = np.array([[-1., -2, -4, -5]])
    actFuns1 = [None, None, None, None]

    link2 = np.array([[0, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0]])
    bia2 = np.array([[-10, -20, -30, -40]])
    actFuns2 = [None, None, None, None]

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 2
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 0, 0, 0])
    assert compare_lists(possible_cuts[1], [0, 3, 3, 0, 0, 0, 0])

def test_possible_cuts_4():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 4), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()],mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.22, 0.33),
                                 r_prob=(0.44, 0.55))
    link1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0]])
    wei1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 1, 1, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0]])
    bia1 = np.array([[-1., -2, -4, -5, -1., -2, -4, -5]])
    actFuns1 = [None, None, ReLu(), ReLu(), ReLu(), None, None, None]

    link2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0]])
    bia2 = np.array([[-10, -20, -30, -40, -50, -40, -50]])
    actFuns2 = [None, None, TanH(), TanH(), None, None, None]

    cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5,
                   mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 12
    assert compare_lists(possible_cuts[0],  [0, 2, 2, 0, 3, 0, 2])
    assert compare_lists(possible_cuts[1],  [0, 2, 3, 0, 3, 1, 1])
    assert compare_lists(possible_cuts[2],  [0, 3, 2, 1, 2, 0, 2])
    assert compare_lists(possible_cuts[3],  [0, 3, 3, 1, 2, 1, 1])
    assert compare_lists(possible_cuts[4],  [0, 3, 4, 1, 2, 2, 0])
    assert compare_lists(possible_cuts[5],  [0, 4, 2, 2, 1, 0, 2])
    assert compare_lists(possible_cuts[6],  [0, 4, 3, 2, 1, 1, 1])
    assert compare_lists(possible_cuts[7],  [0, 4, 4, 2, 1, 2, 0])
    assert compare_lists(possible_cuts[8],  [0, 5, 3, 3, 0, 1, 1])
    assert compare_lists(possible_cuts[9],  [0, 5, 4, 3, 0, 2, 0])
    assert compare_lists(possible_cuts[10], [0, 6, 5, 3, 0, 2, 0])
    assert compare_lists(possible_cuts[11], [0, 7, 6, 3, 0, 2, 0])

def test_gaussian_shift():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = gaussian_shift(weights, links, 0.75, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1.50308048, 1.96986593],
                                                [0, 0, -0.90477989],
                                                [0, 0, 0]]), atol=1e-4))


    seed = 1004
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = gaussian_shift(weights, links, 0.75, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1, 5.72960772],
                                                [0, 0, 6.10237258],
                                                [0, 0, 0]]), atol=1e-4))

def test_reroll_matrix():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = reroll_matrix(weights, links, 0.6, -1, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1.56070304, -0.87660839],
                                                [0, 0, -0.67269435],
                                                [0, 0, 0]]), atol=1e-4))



    seed = 1004
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = reroll_matrix(weights, links, 0.6, -1, 2)

    assert np.all(np.isclose(weights, np.array([[0, 1., 2],
                                                [0, 0, 0.31546711],
                                                [0, 0, 0]]), atol=1e-4))

def test_conditional_try_differnt():
    set = [ReLu(), Poly2(), SincAct(), Poly3()]

    random.seed(1001)

    f1 = conditional_try_choose_different(0.5, ReLu(), set)
    assert f1.to_string() == "RL"

    f1 = conditional_try_choose_different(0.5, ReLu(), set)
    assert f1.to_string() == "P2"

    f1 = conditional_try_choose_different(0.5, SincAct(), set)
    assert f1.to_string() == "SC"

    f1 = conditional_try_choose_different(0.5, Poly3(), set)
    assert f1.to_string() == "P2"

def test_reroll_value():
    random.seed(1002)

    d = 1
    d = reroll_value(0.5, d, -1, 2)
    assert d == 1
    d = 2
    d = reroll_value(0.5, d, -1, 2)
    assert d == pytest.approx(-0.33716, abs=1e-4)
    d = 3
    d = reroll_value(0.5, d, -1, 2)
    assert d == pytest.approx(0.4518733, abs=1e-4)

def test_conditional_value_swap():
    random.seed(1006)

    a = 10
    b = 20
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 20
    assert d == 10

    a = 20
    b = 30
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 30
    assert d == 20

    a = 30
    b = 40
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 40
    assert d == 30

    a = 40
    b = 50
    c, d = conditional_value_swap(0.5, a, b)
    assert c == 40
    assert d == 50

def test_uniform_shift():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = uniform_shift(weights, links, 0.6, -2, 1)

    assert np.all(np.isclose(weights, np.array([[0, 1.56070304, 0.12339161],
                                                [0, 0, 2.32730565],
                                                [0, 0, 0]]), atol=1e-4))


    seed = 1004
    random.seed(seed)
    np.random.seed(seed)


    weights = np.array([[0, 1, 2],
                        [0, 0, 4],
                        [0, 0, 0.]])

    links = np.array([[0, 1, 1],
                      [0, 0, 1],
                      [0, 0, 0]])

    weights = uniform_shift(weights, links, 0.6, -2, 1)

    assert np.all(np.isclose(weights, np.array([[0, 1, 2],
                                                [0, 0, 3.31546711],
                                                [0, 0, 0]]), atol=1e-4))


# seed = 1004
# random.seed(seed)
# np.random.seed(seed)
#
# weights = np.array([[0, 1, 2],
#                     [0, 0, 4],
#                     [0, 0, 0.]])
#
# links = np.array([[0, 1, 1],
#                   [0, 0, 1],
#                   [0, 0, 0]])
#
# probs = np.random.random((3, 3))
# cmask = np.zeros((3, 3))
# cmask[np.where(probs <= 0.6)] = 1
# shift = np.random.uniform(-2, 1, (3, 3))
# shift = np.multiply(shift, cmask)
# w2 = weights.copy()
# w2 += shift
# w2 = np.multiply(w2, links)
# print(w2)
# test_uniform_shift()

# random.seed(1006)
# d = 0.5
# prob = random.random()
# if prob <= d:
#     print("0 ")
# prob = random.random()
# if prob <= d:
#     print("1 ")
# prob = random.random()
# if prob <= d:
#     print("2 ")
# prob = random.random()
# if prob <= d:
#     print("3 ")
# test_conditional_value_swap()

# random.seed(1002)
# d = 0.5
# prob = random.random()
# if prob <= d:
#     print("0 " + str(random.uniform(-1, 2)))
# prob = random.random()
# if prob <= d:
#     print("1 " + str(random.uniform(-1, 2)))
# prob = random.random()
# if prob <= d:
#     print("2 " + str(random.uniform(-1, 2)))
# test_reroll_value()

# random.seed(1001)
# d = 0.5
# set = [ReLu(), Poly2(), SincAct(), Poly3()]
#
# prob = random.random()
# if prob <= d:
#     print("0 " + try_choose_different(ReLu(), set).to_string())
#
# prob = random.random()
# if prob <= d:
#     print("1 " + try_choose_different(ReLu(), set).to_string())
#
# prob = random.random()
# if prob <= d:
#     print("2 " + try_choose_different(SincAct(), set).to_string())
#
# prob = random.random()
# if prob <= d:
#     print("3 " + try_choose_different(Poly3(), set).to_string())
#
# test_conditional_try_differnt()

# seed = 1004
# random.seed(seed)
# np.random.seed(seed)
#
# weights = np.array([[0, 1, 2],
#                     [0, 0, 4],
#                     [0, 0, 0.]])
#
# links = np.array([[0, 1, 1],
#                   [0, 0, 1],
#                   [0, 0, 0]])
#
# probs = np.random.random((3, 3))
# w2 = weights.copy()
# w2[np.where(probs <= 0.6)] = np.random.uniform(-1, 2, (3, 3))[np.where(probs <= 0.6)]
# w2 = np.multiply(w2, links)
# print(w2)
#
# test_reroll_matrix()

# seed = 1004
# random.seed(seed)
# np.random.seed(seed)
#
# weights = np.array([[0, 1, 2],
#                     [0, 0, 4],
#                     [0, 0, 0.]])
#
# links = np.array([[0, 1, 1],
#                   [0, 0, 1],
#                   [0, 0, 0]])
#
# probs = np.random.random((3, 3))
# cmask = np.zeros((3, 3))
# cmask[np.where(probs <= 0.75)] = 1
# shift = np.random.normal(0, 2, (3, 3))
# shift = np.multiply(shift, cmask)
# w2 = weights.copy()
# w2 += shift
# w2 = np.multiply(w2, links)
# print(w2)
# test_gaussian_shift()



# test_possible_cuts_1()
# test_possible_cuts_1_2()
# test_possible_cuts_2()
# test_possible_cuts_3()
# test_possible_cuts_4()

# np.random.seed(1001)
# random.seed(1001)
# density = random.random()
# link_prob = np.random.random((7, 7))
# links = np.zeros((7, 7))
# links[link_prob <= density] = 1
# print(f"links: \n {links}")
# prop_link = np.array([[0., 1., 1., 1., 1., 0., 1.],
#                       [0., 0., 1., 0., 1., 1., 1.],
#                       [0., 1., 0., 1., 0., 0., 1.],
#                       [0., 1., 1., 0., 0., 1., 1.],
#                       [0., 0., 1., 1., 0., 1., 1.],
#                       [0., 0., 0., 0., 0., 0., 0.],
#                       [0., 0., 0., 0., 0., 0., 0.]])
# weights = np.multiply(np.random.uniform(0, 8, (7, 7)), prop_link)
# print(f"weights_rows: \n {weights[3:5, :]}")
# print(f"weights_cols: \n {weights[:, 3:5]}")
# print(f"bias: \n {np.random.uniform(-5, 0, (1, 7))}")
# print(f"af_3: \n {random.randint(0, 3)}")
# print(f"af_4: \n {random.randint(0, 3)}")
#
# np.random.seed(1001)
# random.seed(1001)
# print(f"to_remove: \n{choose_without_repetition([1, 2, 3], 2)}")

# test_neuron_decrease()

# test_network_inflate()
# test_network_deflate()

