from statistics import mean


from evolving_classifier.EvolvingClassifier import EvolvingClassifier, network_from_point
from utility.Utility import *

if __name__ == '__main__':
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_y = one_hot_endode(train_y)
    test_y = one_hot_endode(test_y)

    train_X = [x.reshape(-1, 1) for x in train_X]
    test_X = [x.reshape(-1, 1) for x in test_X]

    hm = 200
    train_X = train_X[:hm]
    train_y = train_y[:hm]
    test_X = test_X[:hm]
    test_y = test_y[:hm]

    ec = EvolvingClassifier()
    ec.hrange.neuronCountMax = 7
    ec.hrange.batchSizeMin = -6
    ec.prepare(20, 20, 0.8, 0.02, 2, (train_X, train_y, test_X, test_y), 1001)
    npoint = ec.run(20, 1)
    network = network_from_point(npoint, 1001)
    network.train(train_X, train_y, 20)
    print(npoint.to_string())
    tests = network.test(test_X, test_y)
    print(tests)
    print(mean(tests[:3]))