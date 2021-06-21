from utility.Mut_Utility import increase_neuron_count, change_neuron_count
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

test_neuron_decrease()

