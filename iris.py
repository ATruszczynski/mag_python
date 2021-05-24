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
    x = iris.data
    y = iris.target

    x = [x.reshape((4, 1)) for x in x]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    x = [x[i] for i in perm]
    y = [y[i] for i in perm]

    train_x = [x[i] for i in range(120)]
    train_y = [y[i] for i in range(120)]
    test_x = [x[i] for i in range(120, 150)]
    test_y = [y[i] for i in range(120, 150)]

    # print(X)
    print(y)

    ec = EvolvingClassifier()
    ec.hrange.neuronCount = 50
    ec.prepare(100, 100, 0.8, 0.02, 2, (train_x, train_y, test_x, test_y), 1001)
    network = ec.run(50, 12)
    tests = network.test(test_x, test_y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    print(network.links)
    print(np.round(network.weights, 2))


    ori = 1

