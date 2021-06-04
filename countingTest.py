from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import OnlyFitnessCalculator
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

    count_tr = 1000
    count_test = 500
    size = 5
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)

    ec = EvolvingClassifier()
    ec.hrange.hiddenLayerCountMin = 0
    ec.hrange.hiddenLayerCountMax = 2
    ec.hrange.neuronCountMax = 50
    ec.co = SimpleCrossoverOperator()
    ec.mo = SimpleMutationOperator(ec.hrange)
    ec.so = TournamentSelection(4)
    ec.ff = ProgressFF(2)
    ec.fc = OnlyFitnessCalculator([1, 0.6, 0.4, 0.25, 0.15, 0.1])
    ec.prepare(popSize=10, startPopSize=10, nn_data=(x, y), seed=1524)
    npoint = ec.run(iterations=1, pm=0.05, pc=0.8, power=12)
    network = network_from_point(npoint, 1001)
    network.train(x, y, 30)
    print(npoint.to_string())
    tests = network.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

