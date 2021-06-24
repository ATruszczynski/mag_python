from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier, CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
from evolving_classifier.operators.HillClimbOperator import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
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

    train_x = [x[i] for i in range(125)]
    train_y = [y[i] for i in range(125)]
    test_x = [x[i] for i in range(125, 150)]
    test_y = [y[i] for i in range(125, 150)]

    # print(X)
    # print(y)

    random.seed(1001)
    np.random.seed(1001)

    ec = EvolvingClassifier()
    # ec.hrange.hiddenLayerCountMin = 0
    # ec.hrange.hiddenLayerCountMax = 0
    # ec.hrange.neuronCountMax = 10
    ec.co = SimpleCrossoverOperator(ec.hrange)
    ec.mo = SimpleAndStructuralCNMutation(ec.hrange, 5)
    ec.so = TournamentSelection(2)
    ec.ff = CNFF()
    ec.fc = CNFitnessCalculator()
    ec.hrange.min_hidden = 10
    ec.hrange.max_hidden = 20
    ec.hco = HillClimbBackpropMutationOperator(1, 100, train_x, train_y)
    ec.prepare(250, 250, (train_x, train_y, test_x, test_y), 50, 1542)
    network = ec.run(iterations=200, pm=0.02, pc=0.8, power=12)
    print(network.links)
    print(network.weights)
    print(network.maxit)
    tests = network.test(test_x, test_y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))


    ori = 1

