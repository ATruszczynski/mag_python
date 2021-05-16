from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    random.seed(1001)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = [x.reshape((4, 1)) for x in X]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    X = [X[i] for i in perm]
    y = [y[i] for i in perm]

    # print(X)
    print(y)

    ec = EvolvingClassifier()
    ec.hrange.neuronCountMax = 6
    ec.prepare(20, 20, 0.8, 0.02, 2, (X, y, X, y), 1001)
    npoint = ec.run(10, 12)
    network = network_from_point(npoint, 1001)
    network.train(X, y, 20, 25)
    print(npoint.to_string())
    tests = network.test(X, y)
    print(network.test(X, y))
    print(mean(tests[:3]))


    ori = 1

