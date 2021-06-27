from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
from evolving_classifier.operators.HillClimbOperator import HillClimbMutationOperator
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    random.seed(1001)
    np.random.seed(1001)

    # sm = Softmax()
    # print(sm.prec_der(np.array([[1], [2], [3]])))
    # print(sm.computeDer(np.array([[1], [2], [3]])))

    count_tr = 2000
    count_test = 500
    size = 5
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)
    #TODO add sized calculator
    #TODO ec better constructor
    ec = EvolvingClassifier()
    # ec.hrange.hiddenLayerCountMin = 0
    # ec.hrange.hiddenLayerCountMax = 0
    # ec.hrange.neuronCountMax = 10
    ec.hrange.max_hidden = 10
    ec.hrange.min_hidden = 5
    ec.co = SimpleCrossoverOperator(ec.hrange)
    ec.mo = SimpleCNMutation(ec.hrange)
    ec.so = TournamentSelection(2)
    ec.ff = CNFF2(QuadDiff())
    # ec.ff = CNFF()
    ec.fc = CNFitnessCalculator()
    # ec.fc = OnlyFitnessCalculator([1, 0.6, 0.4, 0.25, 0.15, 0.1])
    ec.prepare(50, 50, (x, y), 10, 1542)
    network = ec.run(iterations=500, pm=0.05, pc=0.8, power=12)
    # [len(np.where(np.all(ec.population[i].links == 0 and ec.population[i].weights != 0))[0]) for i in range(len(ec.population))]

    print(network.links)
    print(network.weights)
    print(network.maxit)
    tests = network.test(X, Y)
    print([np.sum(ec.population[i].links) for i in range(len(ec.population))])
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

