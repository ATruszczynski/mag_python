import numpy as np
from ann_point.Functions import *
from ann_point.AnnPoint2 import *
from evolving_classifier.operators.CrossoverOperator import *
# from utility.Mut_Utility import resize_layer
from utility.TestingUtility import compare_chaos_network


def test_simple_crossover():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))

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
                   aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
                   aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30)

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
                          desired_s_prob=3)
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
                          desired_s_prob=30)

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
                          desired_s_prob=3)

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
                          desired_s_prob=30)


def test_simple_crossover_2():
    #TODO fix with it changes

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                                 wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))

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

    cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30)

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
                          desired_s_prob=3)

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
                          desired_s_prob=30)

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
                          desired_s_prob=3)

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
                          desired_s_prob=30)


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
                             wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7))
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
               maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3)
cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(),
               maxit=5, mutation_radius=4, wb_mutation_prob=5, s_mutation_prob=6)


random.seed(1002)
poss_cuts = find_possible_cuts(cn1, cn2, hrange)
print(f"cut: {poss_cuts[random.randint(0, len(poss_cuts) - 1)]}")
print(f"prob_swap_aggr: {random.random()}")
print(f"prob_swap_maxit: {random.random()}")
print(f"swap_mut_rad: \n {random.random()}")
print(f"swap_wb_prob: \n {random.random()}")
print(f"swap_s_prob: \n {random.random()}")
test_simple_crossover()























# def test_simple_crossover():
#     pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), TanH()], lossFun=QuadDiff(), learningRate=-1,
#                       momCoeff=-2, batchSize=-3)
#     pointB = AnnPoint(neuronCounts=[2, 30, 5], actFuns=[TanH(), Softmax()], lossFun=CrossEntropy(), learningRate=-10,
#                       momCoeff=-20, batchSize=-30)
#
#     random.seed(1001)
#     co = SimpleCrossoverOperator()
#
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert len(pointC.neuronCounts) == 4
#     assert pointC.neuronCounts[0] == 2
#     assert pointC.neuronCounts[1] == 3
#     assert pointC.neuronCounts[2] == 4
#     assert pointC.neuronCounts[3] == 5
#
#     assert len(pointC.actFuns) == 3
#     assert pointC.actFuns[0].to_string() == ReLu().to_string()
#     assert pointC.actFuns[1].to_string() == Sigmoid().to_string()
#     assert pointC.actFuns[2].to_string() == TanH().to_string()
#
#     assert pointC.lossFun.to_string() == CrossEntropy().to_string()
#
#     assert pointC.learningRate == -1
#     assert pointC.momCoeff == -20
#     assert pointC.batchSize == -30
#
#
#
#     assert len(pointD.neuronCounts) == 3
#     assert pointD.neuronCounts[0] == 2
#     assert pointD.neuronCounts[1] == 30
#     assert pointD.neuronCounts[2] == 5
#
#     assert len(pointD.actFuns) == 2
#     assert pointD.actFuns[0].to_string() == TanH().to_string()
#     assert pointD.actFuns[1].to_string() == Softmax().to_string()
#
#     assert pointD.lossFun.to_string() == QuadDiff().to_string()
#
#     assert pointD.learningRate == -10
#     assert pointD.momCoeff == -2
#     assert pointD.batchSize == -3
#
# def test_layer_swap_crossover():
#     pointA = AnnPoint(neuronCounts=[2, 3, 4, 5], actFuns=[ReLu(), Sigmoid(), TanH()], lossFun=QuadDiff(), learningRate=-1,
#                       momCoeff=-2, batchSize=-3)
#     pointB = AnnPoint(neuronCounts=[2, 30, 5], actFuns=[TanH(), Softmax()], lossFun=CrossEntropy(), learningRate=-10,
#                       momCoeff=-20, batchSize=-30)
#
#     random.seed(1003)
#     co = LayerSwapCrossoverOperator()
#
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert len(pointC.neuronCounts) == 3
#     assert pointC.neuronCounts[0] == 2
#     assert pointC.neuronCounts[1] == 30
#     assert pointC.neuronCounts[2] == 5
#
#     assert len(pointC.actFuns) == 2
#     assert pointC.actFuns[0].to_string() == TanH().to_string()
#     assert pointC.actFuns[1].to_string() == Softmax().to_string()
#
#     assert pointC.lossFun.to_string() == CrossEntropy().to_string()
#
#     assert pointC.learningRate == -1
#     assert pointC.momCoeff == -20
#     assert pointC.batchSize == -30
#
#
#
#     assert len(pointD.neuronCounts) == 4
#     assert pointD.neuronCounts[0] == 2
#     assert pointD.neuronCounts[1] == 3
#     assert pointD.neuronCounts[2] == 4
#     assert pointD.neuronCounts[3] == 5
#
#     assert len(pointD.actFuns) == 3
#     assert pointD.actFuns[0].to_string() == ReLu().to_string()
#     assert pointD.actFuns[1].to_string() == Sigmoid().to_string()
#     assert pointD.actFuns[2].to_string() == TanH().to_string()
#
#     assert pointD.lossFun.to_string() == QuadDiff().to_string()
#
#     assert pointD.learningRate == -10
#     assert pointD.momCoeff == -2
#     assert pointD.batchSize == -3

# def test_crossover():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
#     acts = [ReLu(), ReLu(), Sigmoid(), Sigmoid()]
#     hlc = [2, 1, 2]
#     pointA = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011010)
#     np.random.seed(10011010)
#     co = SomeCrossoverOperator()
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert pointC.input_size == 2
#     assert pointC.output_size == 1
#
#     assert len(pointC.hidden_neuron_counts) == 3
#     assert pointC.hidden_neuron_counts[0] == 2
#     assert pointC.hidden_neuron_counts[1] == 1
#     assert pointC.hidden_neuron_counts[2] == 2
#
#     assert len(pointC.weights) == 4
#     assert np.all(np.isclose(pointC.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[1], np.array([[5, 6.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[2], np.array([[7], [8.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[3], np.array([[-1.51590294, -1.07834955]]), atol=1e-3))
#
#     assert len(pointC.biases) == 4
#     assert np.all(np.isclose(pointC.biases[0], np.array([[-1], [-2]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[1], np.array([[-3]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[2], np.array([[-4], [-5]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[3], np.array([[0]]), atol=1e-3))
#
#     assert len(pointC.activation_functions) == 4
#     assert pointC.activation_functions[0].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[1].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[2].to_string() == Sigmoid().to_string()
#     assert pointC.activation_functions[3].to_string() == Softmax().to_string()
#
#
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     assert pointD.input_size == 2
#     assert pointD.output_size == 1
#
#     assert len(pointD.hidden_neuron_counts) == 1
#     assert pointD.hidden_neuron_counts[0] == 3
#
#     assert len(pointD.weights) == 2
#     assert np.all(np.isclose(pointD.weights[0], np.array([[10, 20], [30, 40.0], [50, 60]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.weights[1], np.array([[0.73125696, 0.60992576, -1.12680355]]), atol=1e-3))
#
#     assert len(pointD.biases) == 2
#     assert np.all(np.isclose(pointD.biases[0], np.array([[-10], [-20.], [-30]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.biases[1], np.array([[0]]), atol=1e-3))
#
#     assert len(pointD.activation_functions) == 2
#     assert pointD.activation_functions[0].to_string() == TanH().to_string()
#     assert pointD.activation_functions[1].to_string() == Sigmoid().to_string()
#
# def test_minimal_damage_crossover():
#     wei = [np.array([[1, 2], [3, 4.0]]), np.array([[5, 6.0]]), np.array([[7], [8.0]]), np.array([[9, 10.0]])]
#     bias = [np.array([[-1], [-2.0]]), np.array([[-3.]]), np.array([[-4.], [-5]]), np.array([[-6.]])]
#     acts = [ReLu(), ReLu(), Sigmoid(), Sigmoid()]
#     hlc = [2, 1, 2]
#     pointA = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     random.seed(10011010)
#     np.random.seed(10011010)
#     co = MinimalDamageCrossoverOperator()
#     pointC, pointD = co.crossover(pointA, pointB)
#
#     assert pointC.input_size == 2
#     assert pointC.output_size == 1
#
#     assert len(pointC.hidden_neuron_counts) == 3
#     assert pointC.hidden_neuron_counts[0] == 2
#     assert pointC.hidden_neuron_counts[1] == 1
#     assert pointC.hidden_neuron_counts[2] == 2
#
#     assert len(pointC.weights) == 4
#     assert np.all(np.isclose(pointC.weights[0], np.array([[1, 2], [3, 4.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[1], np.array([[5, 6.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[2], np.array([[7], [8.0]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.weights[3], np.array([[90, 100.0]]), atol=1e-3))
#
#     assert len(pointC.biases) == 4
#     assert np.all(np.isclose(pointC.biases[0], np.array([[-1], [-2]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[1], np.array([[-3]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[2], np.array([[-4], [-5]]), atol=1e-3))
#     assert np.all(np.isclose(pointC.biases[3], np.array([[-60]]), atol=1e-3))
#
#     assert len(pointC.activation_functions) == 4
#     assert pointC.activation_functions[0].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[1].to_string() == ReLu().to_string()
#     assert pointC.activation_functions[2].to_string() == Sigmoid().to_string()
#     assert pointC.activation_functions[3].to_string() == Softmax().to_string()
#
#
#
#     wei = [np.array([[10, 20], [30, 40.0], [50, 60]]), np.array([[90, 100.0, 110]])]
#     bias = [np.array([[-10], [-20.0], [-30.0]]), np.array([[-60.]])]
#     acts = [TanH(), Softmax()]
#     hlc = [3]
#     pointB = AnnPoint2(2, 1, hlc, acts, wei, bias)
#
#     assert pointD.input_size == 2
#     assert pointD.output_size == 1
#
#     assert len(pointD.hidden_neuron_counts) == 1
#     assert pointD.hidden_neuron_counts[0] == 3
#
#     assert len(pointD.weights) == 2
#     assert np.all(np.isclose(pointD.weights[0], np.array([[10, 20], [30, 40.0], [50, 60]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.weights[1], np.array([[9, 10, 0.0]]), atol=1e-3))
#
#     assert len(pointD.biases) == 2
#     assert np.all(np.isclose(pointD.biases[0], np.array([[-10], [-20.], [-30]]), atol=1e-3))
#     assert np.all(np.isclose(pointD.biases[1], np.array([[-6]]), atol=1e-3))
#
#     assert len(pointD.activation_functions) == 2
#     assert pointD.activation_functions[0].to_string() == TanH().to_string()
#     assert pointD.activation_functions[1].to_string() == Sigmoid().to_string()
#
# def test_layer_resize():
#    layer = resize_layer((3, 3), [-1, 2, ReLu(), np.array([[1, 2], [3, 4]]), np.array([[-1], [-2]])])
#
#    assert layer[0] == -1
#    assert layer[1] == 3
#    assert layer[2].to_string() == ReLu().to_string()
#    assert np.array_equal(layer[3], np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
#    assert np.array_equal(layer[4], np.array([[-1], [-2], [0]]))
#
# def test_layer_resize2():
#     layer = resize_layer((1, 1), [-1, 2, ReLu(), np.array([[1, 2], [3, 4]]), np.array([[-1], [-2]])])
#
#     assert layer[0] == -1
#     assert layer[1] == 1
#     assert layer[2].to_string() == ReLu().to_string()
#     assert np.array_equal(layer[3], np.array([[1]]))
#     assert np.array_equal(layer[4], np.array([[-1]]))

# random.seed(10011010)
# np.random.seed(10011010)
# print(random.randint(0, 3))
# print(get_Xu_matrix((1, 2)))
# print(get_Xu_matrix((1, 3)))
# test_crossover()


# test_minimal_damage_crossover()
#
# random.seed(1003)
# # print(random.randint(0, 5))
# print(random.random())
# print(random.random())
# print(random.random())
# print(random.random())
# print(random.random())

# test_layer_swap_crossover()
#
# seed = 1002
# random.seed(seed)
# np.random.seed(seed)
# print(f"div: \n {random.randint(1, 4)}")
# print(f"swap_1: \n {random.random()}")
# print(f"swap_2: \n {random.random()}")
#
# #
# seed = 1006
# random.seed(seed)
# np.random.seed(seed)
# print(f"div: \n {random.randint(1, 4)}")
# print(f"swap_aggr: \n {random.random()}")
# print(f"swap_max_it: \n {random.random()}")
# print(f"swap_mut_rad: \n {random.random()}")
# print(f"swap_wb_prob: \n {random.random()}")
# print(f"swap_s_prob: \n {random.random()}")
# test_simple_crossover()
# test_simple_crossover_2()