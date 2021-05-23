from statistics import mean


from evolving_classifier.EvolvingClassifier import EvolvingClassifier, network_from_point
from utility.Utility import *

if __name__ == '__main__':

    sm = Softmax()
    print(sm.compute(np.array(-1000)))
    print(sm.compute(np.array(1000)))

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

    ec = EvolvingClassifier()
    ec.hrange.neuronCountMax = 8
    ec.hrange.batchSizeMin = -6
    # TODO pousuwać zbędne argumenty z prepare
    ec.prepare(24, 24, 0.8, 0.05, 2, (train_X, train_y, test_X, test_y), 1001)
    npoint = ec.run(20, 12)
    network = network_from_point(npoint, 1001)
    network.train(train_X, train_y, 50)
    print(npoint.to_string())
    tests = network.test(test_X, test_y)
    print(tests)
    print(mean(tests[:3]))