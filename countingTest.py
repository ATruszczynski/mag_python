from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO1 import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from utility.Utility import *
# from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    seed = 22223333
    random.seed(seed)
    np.random.seed(seed)

    # sm = Softmax()
    # print(sm.prec_der(np.array([[1], [2], [3]])))
    # print(sm.computeDer(np.array([[1], [2], [3]])))

    count_tr = 500
    count_test = 500
    size = 3
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)

    # x, y = generate_square_problem(1000, -2, 2)
    # X, Y = generate_square_problem(100, -1, 1)

    #TODO - B - add sized calculator

    ii = 200

    ec = EvolvingClassifier()
    ec.prepare(popSize=ii, nn_data=(x, y), seed=1524, hrange=get_default_hrange())
    ec.hrange = HyperparameterRange((-10, 10), (-10, 10), (1, 10), (0, 0), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                    mut_radius=(0.001, 1), sqr_mut_prob=(0.001, 1), lin_mut_prob=(0.001, 1), p_mutation_prob=(0.01, 1), c_prob=(0.2, 1),
                                    dstr_mut_prob=(0, 1))
    # ec.co = SimpleCrossoverOperator(ec.hrange)
    # ec.mo = SimpleAndStructuralCNMutation(ec.hrange, 2)
    ec.co = FinalCO1(ec.hrange)
    # ec.co = SimpleCrossoverOperatorHorizontal(ec.hrange)
    ec.mo = FinalMutationOperator(ec.hrange)
    ec.so = TournamentSelection(0.01)
    ec.ff = CNFF()
    # ec.ff = CNFF4(CrossEntropy())
    # ec.ff = CNFF2(QuadDiff())
    # ec.ff = CNFF2(ChebyshevLoss())
    ec.ff = CNFF4(QuadDiff())
    ec.fc = CNFitnessCalculator()
    ec.hrange.min_hidden = 0
    ec.hrange.max_hidden = 20
    network = ec.run(iterations=ii, power=12) # TODO - B - po niewielkiej liczbie iteracji wagi wyglądają podejrzanie? Zła generacja?
    tests = network.test(X, Y, ChebyshevLoss())


    # print(network.links)
    # print(network.weights)
    # print(tests[4])
    # for i in range(11):
    #     x = np.array([[(i - 5.0) / 5.0]])
    #     y = x ** 2
    #     print(f"{x} = {y} - {network.run(x)}")


    # print(network.weights[:6, 5:])
    # print(network.biases[0, 6:])
    # print(network.aggrFun)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

