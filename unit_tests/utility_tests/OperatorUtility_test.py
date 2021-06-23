from evolving_classifier.operators.CrossoverOperator import find_possible_cuts
from utility.Mut_Utility import *
from utility.Utility import *


def test_neuron_increase():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()])

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
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = change_neuron_count(cn1, hrange, 4)

    assert cn2.input_size == 1
    assert cn2.output_size == 2
    assert cn2.neuron_count == 7
    assert cn2.hidden_start_index == 1
    assert cn2.hidden_end_index == 5
    assert cn2.hidden_count == 4

    assert cn2.maxit == 2
    assert cn2.hidden_comp_order is None

    assert np.array_equal(cn2.links, np.array([[0., 1., 1., 1., 1., 0., 1.],
                                               [0., 0., 1., 0., 1., 1., 1.],
                                               [0., 1., 0., 1., 0., 0., 1.],
                                               [0., 1., 1., 0., 0., 1., 1.],
                                               [0., 0., 1., 1., 0., 1., 1.],
                                               [0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0.]]))

    assert np.all(np.isclose(cn2.weights, np.array([[0., 1, 2, 1.99852202, 3.53627653, 0, 4],
                                                    [0 , 0, 3, 0, 2.08113262, 8, 5],
                                                    [0 , 7, 0, 0.87783263, 0, 0, 6],
                                                    [0., 7.15815828, 0.34053276, 0., 0., 5.35134103, 6.56899812],
                                                    [0., 0., 5.58797454, 4.99502278, 0., 7.54306312, 6.34765188],
                                                    [0 , 0, 0, 0, 0, 0, 0],
                                                    [0 , 0, 0, 0, 0, 0, 0]])))

    assert np.all(np.isclose(cn2.biases, np.array([[0., -2, -3, -0.54751109, -1.92308629, -4, -5]])))

    assert np.array_equal(cn2.inp, np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    assert np.array_equal(cn2.act, np.array([[0., 0., 0., 0., 0., 0., 0.]]))

    assert len(cn2.actFuns) == 7
    assert cn2.actFuns[0] is None
    assert cn2.actFuns[1].to_string() == ReLu().to_string()
    assert cn2.actFuns[2].to_string() == Sigmoid().to_string()
    assert cn2.actFuns[3].to_string() == SincAct().to_string()
    assert cn2.actFuns[4].to_string() == ReLu().to_string()
    assert cn2.actFuns[5] is None
    assert cn2.actFuns[6] is None

    assert cn2.aggrFun.to_string() == TanH().to_string()


def test_neuron_decrease():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()])

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
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = change_neuron_count(cn1, hrange, 1)

    assert cn2.input_size == 1
    assert cn2.output_size == 2
    assert cn2.neuron_count == 4
    assert cn2.hidden_start_index == 1
    assert cn2.hidden_end_index == 2
    assert cn2.hidden_count == 1

    assert cn2.maxit == 2
    assert cn2.hidden_comp_order is None

    assert np.array_equal(cn2.links, np.array([[0, 0, 0, 1],
                                               [0, 0, 0, 1],
                                               [0, 0, 0, 0],
                                               [0, 0, 0, 0]]))

    assert np.all(np.isclose(cn2.weights, np.array([[0, 0, 0, 10],
                                                    [0, 0, 0, 13],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0.]])))

    assert np.all(np.isclose(cn2.biases, np.array([[0., -4, -5, -6]])))

    assert np.array_equal(cn2.inp, np.array([[0., 0., 0., 0.]]))
    assert np.array_equal(cn2.act, np.array([[0., 0., 0., 0.]]))

    assert len(cn2.actFuns) == 4
    assert cn2.actFuns[0] is None
    assert cn2.actFuns[1].to_string() == SincAct().to_string()
    assert cn2.actFuns[2] is None
    assert cn2.actFuns[3] is None

    assert cn2.aggrFun.to_string() == TanH().to_string()

def test_network_inflate():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()])

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
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = inflate_network(cn1, 2)

    assert cn2.input_size == 1
    assert cn2.output_size == 2
    assert cn2.neuron_count == 7
    assert cn2.hidden_start_index == 1
    assert cn2.hidden_end_index == 5
    assert cn2.hidden_count == 4

    assert cn2.maxit == 2
    assert cn2.hidden_comp_order is None

    assert np.array_equal(cn2.links, np.array([[0, 1, 0, 0, 0, 0, 1],
                                               [0, 0, 1, 0, 0, 1, 1],
                                               [0, 1, 1, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0]]))

    assert np.all(np.isclose(cn2.weights, np.array([[0, 1, 0, 0, 0, 0, 6],
                                                    [0, 0, 3, 0, 0, 5, 7],
                                                    [0, 2, 4, 0, 0, 0, 8],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]])))

    assert np.all(np.isclose(cn2.biases, np.array([[0., -2, -4, 0, 0, -5, -6]])))

    assert np.array_equal(cn2.inp, np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    assert np.array_equal(cn2.act, np.array([[0., 0., 0., 0., 0., 0., 0.]]))

    assert len(cn2.actFuns) == 7
    assert cn2.actFuns[0] is None
    assert cn2.actFuns[1].to_string() == ReLu().to_string()
    assert cn2.actFuns[2].to_string() == SincAct().to_string()
    assert cn2.actFuns[3].to_string() == ReLu().to_string()
    assert cn2.actFuns[4].to_string() == SincAct().to_string()
    assert cn2.actFuns[5] is None
    assert cn2.actFuns[6] is None

    assert cn2.aggrFun.to_string() == TanH().to_string()

def test_network_deflate():
    hrange = HyperparameterRange((-1, 1), (-10, 10), (0, 5), (0, 5), [SincAct(), ReLu(), Sigmoid(), TanH()])

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
    bia1 = np.array([[0., -2, -4, -5, -6, -7, -8, -9, -10]])
    actFuns1 = [None, None, SincAct(), TanH(), LReLu(), GaussAct(), Sigmoid(), None, None]

    cn1 = ChaosNet(input_size=2, output_size=2, links=link1.copy(), weights=wei1.copy(), biases=bia1.copy(),
                   actFuns=actFuns1, aggrFun=TanH(), maxit=2)

    np.random.seed(1001)
    random.seed(1001)

    cn2 = deflate_network(cn1)

    assert cn2.input_size == 2
    assert cn2.output_size == 2
    assert cn2.neuron_count == 7
    assert cn2.hidden_start_index == 2
    assert cn2.hidden_end_index == 5
    assert cn2.hidden_count == 3

    assert cn2.maxit == 2
    assert cn2.hidden_comp_order is None

    assert np.array_equal(cn2.links, np.array([[0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 1, 0, 1, 0, 1],
                                               [0, 0, 1, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0]]))

    assert np.all(np.isclose(cn2.weights, np.array([[0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 3, 0, 6, 0, 8],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 4, 0, 7, 0, 9],
                                                    [0, 0, 5, 0, 0, 0, 10],
                                                    [0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0]])))

    assert np.all(np.isclose(cn2.biases, np.array([[0., -2, -4, -5, -6, -9, -10]])))

    assert np.array_equal(cn2.inp, np.array([[0., 0., 0., 0., 0., 0., 0.]]))
    assert np.array_equal(cn2.act, np.array([[0., 0., 0., 0., 0., 0., 0.]]))

    assert len(cn2.actFuns) == 7
    assert cn2.actFuns[0] is None
    assert cn2.actFuns[1] is None
    assert cn2.actFuns[2].to_string() == SincAct().to_string()
    assert cn2.actFuns[3].to_string() == TanH().to_string()
    assert cn2.actFuns[4].to_string() == LReLu().to_string()
    assert cn2.actFuns[5] is None
    assert cn2.actFuns[6] is None

    assert cn2.aggrFun.to_string() == TanH().to_string()

def test_possible_cuts_1():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()])
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

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 4
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 2, 0, 0])
    assert compare_lists(possible_cuts[1], [0, 3, 2, 1, 1, 0, 0])
    assert compare_lists(possible_cuts[2], [0, 4, 2, 2, 0, 0, 0])
    assert compare_lists(possible_cuts[3], [0, 5, 3, 2, 0, 0, 0])

def test_possible_cuts_1_2():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()])

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

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 4
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 0, 0, 2])
    assert compare_lists(possible_cuts[1], [0, 2, 3, 0, 0, 1, 1])
    assert compare_lists(possible_cuts[2], [0, 2, 4, 0, 0, 2, 0])
    assert compare_lists(possible_cuts[3], [0, 3, 5, 0, 0, 2, 0])


def test_possible_cuts_2():
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (1, 3), [ReLu(), Sigmoid(), GaussAct(), TanH()])
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

    cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2)
    cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5)

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
                                 hidden_count=(0, 3), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()])
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

    cn1 = ChaosNet(input_size=2, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2)
    cn2 = ChaosNet(input_size=2, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5)

    possible_cuts = find_possible_cuts(cn1, cn2, hrange)
    print(possible_cuts)

    assert len(possible_cuts) == 2
    assert compare_lists(possible_cuts[0], [0, 2, 2, 0, 0, 0, 0])
    assert compare_lists(possible_cuts[1], [0, 3, 3, 0, 0, 0, 0])

def test_possible_cuts_4():
    hrange = HyperparameterRange(init_wei=(-1, 1), init_bia=(-1, 1), it=(1, 5),
                                 hidden_count=(0, 4), actFuns=[ReLu(), Sigmoid(), GaussAct(), TanH()])
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

    cn1 = ChaosNet(input_size=2, output_size=3, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1, aggrFun=SincAct(), maxit=2)
    cn2 = ChaosNet(input_size=2, output_size=3, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2, aggrFun=GaussAct(), maxit=5)

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


test_network_deflate()

