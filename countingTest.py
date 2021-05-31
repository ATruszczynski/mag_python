from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
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
    ec.hrange.layerCountMin = 0
    ec.hrange.layerCountMax = 1
    ec.hrange.neuronCountMax = 25
    ec.sco = MinimalDamageCrossoverOperator()
    ec.co = WBCrossoverOperator()
    ec.smo = SomeStructMutationOperator(ec.hrange)
    ec.mo = BiasedGaussianWBMutationOperator(ec.hrange)
    ec.so = TournamentSelection(4)
    ec.ff = CrossEffFitnessFunction3()
    ec.hco = HillClimbMutationOperator(1, 5, ec.mo)
    ec.prepare(popSize=5, startPopSize=5, nn_data=(x, y, X, Y), seed=1524)
    npoint = ec.run(iterations=70, finetune=30, pm=0.75, pms=0.05, pc=0.75, power=12)
    network = network_from_point(npoint, 1001)
    print(npoint.to_string())
    tests = network.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

