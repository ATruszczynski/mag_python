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

    count_tr = 200
    count_test = 500
    size = 3
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)
    #TODO add sized calculator
    #TODO ec better constructor
    ec = EvolvingClassifier()
    ec.hrange = HyperparameterRange((-10, 10), (-10, 10), (1, 10), (0, 0), [ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                    mut_radius=(-5, 5), wb_mut_prob=(0, 1), s_mut_prob=(0, 1), p_mutation_prob=(0.01, 1))
    # ec.co = SimpleCrossoverOperator(ec.hrange)
    # ec.mo = SimpleAndStructuralCNMutation(ec.hrange, 2)
    ec.co = SimpleCrossoverOperator(ec.hrange)
    ec.mo = TestMutationOperator(ec.hrange)
    ec.so = TournamentSelection(2)
    ec.ff = CNFF()
    # ec.ff = CNFF4(CrossEntropy())
    # ec.ff = CNFF2(QuadDiff())
    # ec.ff = CNFF2(ChebyshevLoss())
    # ec.ff = CNFF4(QuadDiff())
    ec.fc = CNFitnessCalculator()
    ec.hrange.min_hidden = 0
    ec.hrange.max_hidden = 10
    ec.prepare(popSize=100, startPopSize=100, nn_data=(x, y), hidden_size=10, seed=1524)
    network = ec.run(iterations=1000, pm=0.05, pc=0.0, power=12) # TODO po niewielkiej liczbie iteracji wagi wyglądają podejrzanie? Zła generacja?
    tests = network.test(X, Y, ChebyshevLoss())
    print(network.links)
    print(network.weights[:6, 5:])
    print(network.biases[0, 6:])
    print(network.aggrFun)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

