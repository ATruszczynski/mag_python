from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    random.seed(1001)

    # sm = Softmax()
    # print(sm.prec_der(np.array([[1], [2], [3]])))
    # print(sm.computeDer(np.array([[1], [2], [3]])))

    count = 1000
    size = 10
    x,y = generate_counting_problem(count, size)
    X,Y = generate_counting_problem(ceil(count/5), size)

    ec = EvolvingClassifier()
    ec.hrange.neuronCountMin = 0
    ec.hrange.neuronCountMax = 8
    ec.hrange.layerCountMin = 0
    ec.hrange.layerCountMax = 3
    ec.hrange.batchSizeMin = -6
    ec.prepare(20, 20, 0.8, 0.05, 2, (x, y, X, Y), 1001)
    npoint = ec.run(20, 12)
    network = network_from_point(npoint, 1001)
    network.train(x, y, 50)
    print(npoint.to_string())
    tests = network.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

