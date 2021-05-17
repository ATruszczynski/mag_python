from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':

    # x = np.exp(-10000)
    # print(1/ (1 + x))

    sg = Sigmoid()
    sg.compute(-1000)

    tanh = TanH()
    print(tanh.compute(np.array([-1000])))
    print(tanh.compute(np.array([1000])))
    print(tanh.computeDer(np.array([1000])))
    print(tanh.computeDer(np.array([-1000])))

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
    ec.hrange.neuronCountMax = 2
    ec.hrange.batchSizeMin = -6
    ec.prepare(20, 20, 0.8, 0.02, 2, (X, y, X, y), 1001)
    npoint = ec.run(20, 12)
    network = network_from_point(npoint, 1001)
    network.train(X, y, 20)
    print(npoint.to_string())
    tests = network.test(X, y)
    print(network.test(X, y))
    print(mean(tests[:3]))


    ori = 1

