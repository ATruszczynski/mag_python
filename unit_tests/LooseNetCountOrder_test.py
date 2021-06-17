from ann_point.Functions import ReLu
from neural_network.ChaosNet import ChaosNet
import numpy as np

# def test_cn_comp_order():
#     links = np.array([[0, 0, 0, 1, 0, 0],
#                       [0, 0, 1, 0, 0, 0],
#                       [0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 0, 0, 1],
#                       [0, 0, 0, 0, 0, 1],
#                       [0, 0, 0, 0, 0, 0]])
#     biases = np.array([[0, 0, 0, 0, 0, 0]])
#     cn = ChaosNet(input_size=2, output_size=1, links=links, weights=links, biases=biases, actFuns=6 *[None], aggrFun=ReLu())
#     cn.get_comp_order()
#
#     assert len(cn.hidden_comp_order) == 2
#
#     assert len(cn.hidden_comp_order[0]) == 1
#     assert cn.hidden_comp_order[0][0] == 2
#
#     assert len(cn.hidden_comp_order[1]) == 2
#     assert cn.hidden_comp_order[1][0] == 3
#     assert cn.hidden_comp_order[1][1] == 4
#
# def test_cn_comp_order_2():
#     links = np.array([[0, 0, 1, 0, 0, 0, 0],
#                       [0, 0, 0, 1, 0, 1, 1],
#                       [0, 0, 0, 1, 1, 0, 0],
#                       [0, 0, 0, 0, 0, 1, 0],
#                       [0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0]])
#     biases = np.array([[0, 0, 0, 0, 0, 0, 0]])
#     cn = ChaosNet(input_size=2, output_size=3, links=links, weights=links, biases=biases, actFuns=7 *[None], aggrFun=ReLu())
#     cn.get_comp_order()
#
#     assert len(cn.hidden_comp_order) == 2
#
#     assert len(cn.hidden_comp_order[0]) == 1
#     assert cn.hidden_comp_order[0][0] == 2
#
#     assert len(cn.hidden_comp_order[1]) == 1
#     assert cn.hidden_comp_order[1][0] == 3



def test_cn_comp_order_rec():
    links = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    biases = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])
    cn = ChaosNet(input_size=2, output_size=2, links=links, weights=links, biases=biases, actFuns=9 *[None], aggrFun=ReLu())
    cn.get_comp_order()

    assert len(cn.hidden_comp_order) == 5

    assert cn.hidden_comp_order[0] == 2
    assert cn.hidden_comp_order[1] == 4
    assert cn.hidden_comp_order[2] == 5
    assert cn.hidden_comp_order[3] == 6
    assert cn.hidden_comp_order[4] == 3




# def test_ln_outgoing():
#     links = np.array([[0, 1, 0, 0, 0],
#                       [0, 0, 1, 1, 0],
#                       [0, 0, 0, 0, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0]])
#     biases = np.array([0, 0, 0, 0, 0])
#     cn = ChaosNet(input_size=1, output_size=2, links=links, weights=links, biases=biases, actFuns=5 *[None], aggrFun=ReLu(), maxIt=2)
#
#     touched1 = cn.get_neurons_to_update([0, 1])
#
#     assert len(touched1) == 5
#     assert touched1[0] == 0
#     assert touched1[1] == 1
#     assert touched1[2] == 2
#     assert touched1[3] == 3
#     assert touched1[4] == 4
#
#     touched2 = cn.get_neurons_to_update([2, 4])
#
#     assert len(touched2) == 4
#     assert touched2[0] == 1
#     assert touched2[1] == 2
#     assert touched2[2] == 3
#     assert touched2[3] == 4
#
# def test_cn_outgoing_rec():
#     links = np.array([[0, 1, 0, 0, 0],
#                       [0, 0, 1, 1, 0],
#                       [0, 0, 0, 0, 1],
#                       [0, 0, 0, 0, 0],
#                       [0, 1, 0, 0, 0]])
#     biases = np.array([0, 0, 0, 0, 0])
#     cn = ChaosNet(input_size=1, output_size=2, links=links, weights=links, biases=biases, actFuns=5 *[None], aggrFun=ReLu(), maxIt=2)
#
#     touched1 = cn.get_neurons_to_update_rec(0, 5 * [0])
#
#     assert len(touched1) == 5
#     assert touched1[0] == 0
#     assert touched1[1] == 1
#     assert touched1[2] == 2
#     assert touched1[3] == 4
#     assert touched1[4] == 3
#
#     touched2 = cn.get_neurons_to_update_rec(2, 5 * [0])
#
#     assert len(touched2) == 4
#     assert touched2[0] == 2
#     assert touched2[1] == 4
#     assert touched2[2] == 1
#     assert touched2[3] == 3
#
#
# def test_ln_outgoing_2():
#     links = np.array([[0, 1, 1, 0, 0],
#                       [0, 0, 0, 1, 0],
#                       [0, 0, 0, 0, 1],
#                       [0, 1, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     biases = np.array([0, 0, 0, 0, 0])
#     cn = ChaosNet(input_size=1, output_size=2, links=links, weights=links, biases=biases, actFuns=5 *[None], aggrFun=ReLu(), maxIt=2)
#
#     touched1 = cn.get_neurons_to_update([0])
#
#     assert len(touched1) == 5
#     assert touched1[0] == 0
#     assert touched1[1] == 1
#     assert touched1[2] == 2
#     assert touched1[3] == 3
#     assert touched1[4] == 4
#
#     touched2 = cn.get_neurons_to_update([1])
#
#     assert len(touched2) == 2
#     assert touched2[0] == 1
#     assert touched2[1] == 3
#
#     touched3 = cn.get_neurons_to_update([3, 4])
#
#     assert len(touched3) == 3
#     assert touched3[0] == 1
#     assert touched3[1] == 3
#     assert touched3[2] == 4
#
#     touched4 = cn.get_neurons_to_update([3, 2])
#
#     assert len(touched4) == 4
#     assert touched4[0] == 1
#     assert touched4[1] == 2
#     assert touched4[2] == 3
#     assert touched4[3] == 4
#
# def test_cn_comp_order():
#     links = np.array([[0, 1, 1, 0, 0],
#                       [0, 0, 0, 1, 0],
#                       [0, 0, 0, 0, 1],
#                       [0, 1, 0, 0, 0],
#                       [0, 0, 0, 0, 0]])
#     biases = np.array([0, 0, 0, 0, 0])
#     cn = ChaosNet(input_size=1, output_size=2, links=links, weights=links, biases=biases, actFuns=5 *[None], aggrFun=ReLu(), maxIt=2)
#
#     cn.get_comp_order()
#     ori = 1
#
# test_cn_comp_order()

# test_cn_comp_order()
# test_cn_comp_order_2()
test_cn_comp_order_rec()




