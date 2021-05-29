from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessFunction import CrossEffFitnessFunction
from evolving_classifier.operators.CrossoverOperator import SomeCrossoverOperator
from evolving_classifier.operators.MutationOperators import SomeStructMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    random.seed(1001)

    # sm = Softmax()
    # print(sm.prec_der(np.array([[1], [2], [3]])))
    # print(sm.computeDer(np.array([[1], [2], [3]])))

    count_tr = 1000
    count_test = 500
    size = 10
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)

    ec = EvolvingClassifier()
    ec.hrange.layerCountMin = 0
    ec.hrange.layerCountMax = 2
    ec.co = SomeCrossoverOperator()
    ec.mo = SomeStructMutationOperator(ec.hrange)
    ec.so = TournamentSelection(2)
    ec.ff = CrossEffFitnessFunction()
    ec.prepare(popSize=100, startPopSize=100, nn_data=(x, y, X, Y), seed=1524)
    npoint = ec.run(100, 12)
    network = network_from_point(npoint, 1001)
    print(npoint.to_string())
    tests = network.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

