from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
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

    count_tr = 1000
    count_test = 500
    size = 5
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)
    #TODO add sized calculator
    #TODO ec better constructor
    ec = EvolvingClassifier()
    ec.co = SimpleCrossoverOperator(ec.hrange)
    ec.mo = SimpleAndStructuralCNMutation(ec.hrange, 5)
    ec.so = TournamentSelection(2)
    ec.ff = CNFF()
    # ec.ff = CNFF2(ChebyshevLoss())
    ec.fc = CNFitnessCalculator()
    ec.hrange.min_hidden = 0
    ec.hrange.max_hidden = 0
    ec.prepare(popSize=500, startPopSize=500, nn_data=(x, y), hidden_size=10, seed=1524)
    network = ec.run(iterations=200, pm=0.05, pc=0.8, power=12)
    tests = network.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

