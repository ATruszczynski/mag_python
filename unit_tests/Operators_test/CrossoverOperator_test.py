# import numpy as np
from ann_point.Functions import *
# from ann_point.AnnPoint2 import *
from evolving_classifier.operators.CrossoverOperator import *
# from utility.Mut_Utility import resize_layer
from utility.TestingUtility import compare_chaos_network


def test_simple_crossover():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 r_prob=(0, 0)) # values irrelevant aside from neuron count

    #TODO fix with it changes
    link1 = np.array([[0, 1, 1, 0, 1],
                      [0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 2, 0, 4],
                     [0 , 0, 3, 0, 5],
                     [0 , 7, 0, 0, 6],
                     [0 , 0, 0, 0, 0],
                     [0 , 0, 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4, -5]])
    actFuns1 = [None, ReLu(), ReLu(), None, None]

    link2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
    wei2 = np.array([[0, 0, 0,  0,  0],
                     [0, 0, 10, 20, 0],
                     [0, 0, 0,  30, 40],
                     [0, 0, 0,  0,  0],
                     [0, 0, 0,  0,  0]])
    bia2 = np.array([[0., -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                   aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2,
                   s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                   aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20,
                   s_mutation_prob=30, p_mutation_prob=40, c_prob=50, r_prob=60)

    co = SimpleCrossoverOperator(hrange)

    random.seed(1002)
    np.random.seed(1002)
    cn3, cn4 = co.crossover(cn1, cn2)



    ##################################################################

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
                                                    [0 , 0, 3, 0, 5],
                                                    [0 , 7, 0, 0, 6],
                                                    [0 , 0, 0, 0, 0],
                                                    [0 , 0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -4, -5]]),
                          desired_actFun=[None, ReLu(), ReLu(), None, None],
                          desired_aggr=SincAct(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6)
    #TODO biasy tu są źle
    ##################################################################

    compare_chaos_network(net=cn2,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0,  0,  0],
                                                    [0, 0, 10, 20, 0],
                                                    [0, 0, 0,  30, 40],
                                                    [0, 0, 0,  0,  0],
                                                    [0, 0, 0,  0,  0]]),
                          desired_biases=np.array([[0, -20, -30, -40, -50]]),
                          desired_actFun=[None, TanH(), TanH(), None, None],
                          desired_aggr=GaussAct(),
                          desired_maxit=5,
                          desired_mut_rad=10,
                          desired_wb_prob=20,
                          desired_s_prob=30,
                          desired_p_prob=40,
                          desired_c_prob=50,
                          desired_r_prob=60)

    ##################################################################

    compare_chaos_network(net=cn3,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 1, 0, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 1, 0, 1, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 1, 2, 0, 0],
                                                    [0, 0, 3, 20, 0],
                                                    [0, 7, 0, 30, 40],
                                                    [0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0]]),
                          desired_biases=np.array([[0, -2, -3, -40, -50]]),
                          desired_actFun=[None, ReLu(), ReLu(), None, None],
                          desired_aggr=SincAct(),
                          desired_maxit=5,
                          desired_mut_rad=10,
                          desired_wb_prob=20,
                          desired_s_prob=3,
                          desired_p_prob=40,
                          desired_c_prob=5,
                          desired_r_prob=6)

    ##################################################################

    compare_chaos_network(net=cn4,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 0, 0, 0, 1],
                                                  [0, 0, 1, 0, 1],
                                                  [0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0,  0, 4],
                                                    [0, 0, 10, 0, 5],
                                                    [0, 0, 0,  0, 6],
                                                    [0, 0, 0,  0, 0],
                                                    [0, 0, 0,  0, 0]]),
                          desired_biases=np.array([[0., -20, -30, -4, -5]]),
                          desired_actFun=[None, TanH(), TanH(), None, None],
                          desired_aggr=GaussAct(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=30,
                          desired_p_prob=4,
                          desired_c_prob=50,
                          desired_r_prob=60)


def test_simple_crossover_2():
    #TODO fix with it changes

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                                 r_prob=(0, 0))

    link1 = np.array([[0, 1, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
    wei1 = np.array([[0., 1, 0, 4],
                     [0 , 0, 0, 5],
                     [0 , 0, 0, 0],
                     [0 , 0, 0, 0]])
    bia1 = np.array([[0., -2, -3, -4]])
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
    bia2 = np.array([[0, -20, -30, -40, -50]])
    actFuns2 = [None, TanH(), TanH(), None, None]

    cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
                   aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3,
                   p_mutation_prob=4, c_prob=5, r_prob=6)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                   aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30,
                   p_mutation_prob=40, c_prob=50, r_prob=60)

    co = SimpleCrossoverOperator(hrange)

    seed = 1006
    random.seed(seed)
    np.random.seed(seed)
    cn3, cn4 = co.crossover(cn1, cn2)


    ##################################################################

    compare_chaos_network(net=cn1,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=4,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=2,
                          desired_hidden_count=1,
                          desired_links=np.array([[0, 1, 0, 1],
                                                  [0, 0, 0, 1],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]),
                          desired_weights=np.array([[0., 1, 0, 4],
                                                    [0 , 0, 0, 5],
                                                    [0 , 0, 0, 0],
                                                    [0 , 0, 0, 0]]),
                          desired_biases=np.array([[0., -2, -3, -4]]),
                          desired_actFun=[None, ReLu(), None, None],
                          desired_aggr=SincAct(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6)

    ##################################################################

    compare_chaos_network(net=cn2,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0, 0, 0,  0,  0 ],
                                                    [0, 0, 10, 20, 0 ],
                                                    [0, 0, 0,  30, 40],
                                                    [0, 0, 0,  0,  0 ],
                                                    [0, 0, 0,  0,  0.]]),
                          desired_biases=np.array([[0, -20, -30, -40, -50]]),
                          desired_actFun=[None, TanH(), TanH(), None, None],
                          desired_aggr=GaussAct(),
                          desired_maxit=5,
                          desired_mut_rad=10,
                          desired_wb_prob=20,
                          desired_s_prob=30,
                          desired_p_prob=40,
                          desired_c_prob=50,
                          desired_r_prob=60)

    #TODO biasy tu są źle

    ##################################################################

    compare_chaos_network(net=cn3,
                          desired_input_size=1,
                          desited_output_size=2,
                          desired_neuron_count=5,
                          desired_hidden_start_index=1,
                          desired_hidden_end_index=3,
                          desired_hidden_count=2,
                          desired_links=np.array([[0, 1, 0, 0, 0],
                                                  [0, 0, 1, 1, 0],
                                                  [0, 0, 0, 1, 1],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0]]),
                          desired_weights=np.array([[0., 1, 0,  0,  0 ],
                                                    [0,  0,  10, 20, 0 ],
                                                    [0,  0,  0,  30, 40],
                                                    [0,  0,  0,  0,  0 ],
                                                    [0,  0,  0,  0,  0 ]]),
                          desired_biases=np.array([[0., -2, -30, -40, -50]]),
                          desired_actFun=[None, ReLu(), TanH(), None, None],
                          desired_aggr=GaussAct(),
                          desired_maxit=5,
                          desired_mut_rad=10,
                          desired_wb_prob=2,
                          desired_s_prob=3,
                          desired_p_prob=4,
                          desired_c_prob=5,
                          desired_r_prob=6)

    ###################################################################

    compare_chaos_network(net=cn4,
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
                          desired_weights=np.array([[0, 0, 0, 4],
                                                    [0, 0, 0, 5],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0]]),
                          desired_biases=np.array([[0., -20, -3, -4]]),
                          desired_actFun=[None, TanH(), None, None],
                          desired_aggr=SincAct(),
                          desired_maxit=2,
                          desired_mut_rad=1,
                          desired_wb_prob=20,
                          desired_s_prob=30,
                          desired_p_prob=40,
                          desired_c_prob=50,
                          desired_r_prob=60)

# def test_test_crossover():
#     hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                                  wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6))
#
#     #TODO fix with it changes
#     link1 = np.array([[0, 1, 1, 0, 1],
#                       [0, 0, 1, 0, 1],
#                       [0, 1, 0, 0, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     wei1 = np.array([[0., 1, 2, 0, 4],
#                      [0 , 0, 3, 0, 5],
#                      [0 , 7, 0, 0, 6],
#                      [0 , 0, 0, 0, 0],
#                      [0 , 0, 0, 0, 0]])
#     bia1 = np.array([[0., -2, -3, -4, -5]])
#     actFuns1 = [None, ReLu(), ReLu(), None, None]
#
#     link2 = np.array([[0, 0, 0, 0, 0],
#                       [0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     wei2 = np.array([[0, 0, 0,  0,  0],
#                      [0, 0, 10, 20, 0],
#                      [0, 0, 0,  30, 40],
#                      [0, 0, 0,  0,  0],
#                      [0, 0, 0,  0,  0]])
#     bia2 = np.array([[0., -20, -30, -40, -50]])
#     actFuns2 = [None, TanH(), TanH(), None, None]
#
#     cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
#                    aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4)
#     cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
#                    aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30, p_mutation_prob=40)
#
#     co = TestCrossoverOperator()
#
#     seed = 1001
#     random.seed(seed)
#     np.random.seed(seed)
#     cn3, cn4 = co.crossover(cn1, cn2)
#
#     print(cn3.links)
#     print(cn3.weights)
#     print(cn4.links)
#     print(cn4.weights)

# seed=1001
# random.seed(seed)
#
# cut_ori = random.randint(0, 1)
# print(cut_ori)
# if cut_ori == 0:
#     print(random.randint(2, 4))
# else:
#     print(random.randint(1, 2))




# test_test_crossover()


# link1 = np.array([[0, 1, 0, 1],
#                   [0, 0, 0, 1],
#                   [0, 0, 0, 0],
#                   [0, 0, 0, 0]])
# wei1 = np.array([[0., 1, 0, 4],
#                  [0 , 0, 0, 5],
#                  [0 , 0, 0, 0],
#                  [0 , 0, 0, 0]])
# bia1 = np.array([[-1., -2, -4, -5]])
# actFuns1 = [None, ReLu(), None, None]
#
# link2 = np.array([[0, 0, 0, 0, 0],
#                   [0, 0, 1, 1, 0],
#                   [0, 0, 0, 1, 1],
#                   [0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0]])
# wei2 = np.array([[0, 0, 0,  0,  0 ],
#                  [0, 0, 10, 20, 0 ],
#                  [0, 0, 0,  30, 40],
#                  [0, 0, 0,  0,  0 ],
#                  [0, 0, 0,  0,  0.]])
# bia2 = np.array([[-10, -20, -30, -40, -50]])
# actFuns2 = [None, TanH(), TanH(), None, None]
#
# hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                              wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
#
# cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)
# cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30)



hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                             wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.3, 0.5), c_prob=(0.4, 0.6),
                             r_prob=(0.8, 1))
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
actFuns1 = [None, ReLu(), ReLu(), None, None]

link2 = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
wei2 = np.array([[0, 0, 0, 0, 0],
                 [0, 0, 10, 20, 0],
                 [0, 0, 0, 30, 40],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
bia2 = np.array([[-10, -20, -30, -40, -50]])
actFuns2 = [None, TanH(), TanH(), None, None]

cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(),
               maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3, p_mutation_prob=4, c_prob=5, r_prob=6)
cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(),
               maxit=5, mutation_radius=4, wb_mutation_prob=5, s_mutation_prob=6, p_mutation_prob=7, c_prob=8, r_prob=9)


random.seed(1006)
poss_cuts = find_possible_cuts(cn1, cn2, hrange)
print(f"cut: {poss_cuts[random.randint(0, len(poss_cuts) - 1)]}")
print(f"prob_swap_aggr: {random.random()}")
print(f"prob_swap_maxit: {random.random()}")
print(f"swap_mut_rad: \n {random.random()}")
print(f"swap_wb_prob: \n {random.random()}")
print(f"swap_s_prob: \n {random.random()}")
print(f"swap_p_prob: \n {random.random()}")
print(f"swap_c_prob: \n {random.random()}")
print(f"swap_r_prob: \n {random.random()}")
# test_simple_crossover()
# test_simple_crossover_2()

