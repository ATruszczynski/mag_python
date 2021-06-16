from ann_point.Functions import *

from ann_point.AnnPoint2 import AnnPoint2
from evolving_classifier.operators.MutationOperators import *
import numpy as np
import time


# def compare_mutations():
#     smo = BiasedGaussianWBMutationOperator(get_default_hrange())
#     smo2 = SomeWBMutationOperator2(get_default_hrange())
#
#     point = AnnPoint2(10000, 1000, weights=[np.zeros((1000, 10000))], biases=[np.zeros((1000, 1))], activationFuns=[ReLu()], hiddenNeuronCounts=[])
#
#     n = 100
#
#     s = time.time()
#     for i in range(n):
#         _ = smo2.mutate(point, 1, 2)
#     t = time.time()
#
#     print(round((t - s)/n, 2))
#
#
#     s = time.time()
#     for i in range(n):
#         _ = smo.mutate(point, 1, 2)
#     t = time.time()
#
#     print(round((t - s)/n, 2))
#
# compare_mutations()
from neural_network.ChaosNet import ChaosNet
from neural_network.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork

if __name__ == '__main__':
    #TODO podział na zbiór uczący EC, zbiór uczący NN i zbiór testowy?
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_y = one_hot_endode(train_y)
    test_y = one_hot_endode(test_y)

    train_X = [x.reshape(-1, 1) for x in train_X]
    test_X = [x.reshape(-1, 1) for x in test_X]

    hm = 10000
    train_X = train_X[:hm]
    train_y = train_y[:hm]
    test_X = test_X[:ceil(hm/3)]
    test_y = test_y[:ceil(hm/3)]

    links = np.zeros((794, 794))
    links[:784, 784:] = 1
    wei = np.zeros((794, 794))
    wei[:784, 784:] = 1
    acts = 794 * [ReLu()]

    cn = ChaosNet(input_size=784, output_size=10, links=links, weights=wei, biases=np.zeros((1, 794)), actFuns=acts, aggrFun=ReLu(), maxIt=2)
    ff = FeedForwardNeuralNetwork([784, 10], [ReLu()], QuadDiff(), 1, 1, 1, 1001)
    print("PRM")
    n = 10000
    s = time.time()
    for i in range(n):
        result = cn.run(train_X[i])
    t = time.time()

    print(round((t - s) / n, 7))
    # print(result)

    s = time.time()
    for i in range(n):
        result = ff.run(train_X[i])
    t = time.time()

    print(round((t - s) / n, 7))
    # print(result)