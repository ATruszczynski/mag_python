from statistics import mean


from evolving_classifier.EvolvingClassifier import EvolvingClassifier, network_from_point
from evolving_classifier.FitnessFunction import CrossEffFitnessFunction
from evolving_classifier.operators.CrossoverOperator import SomeCrossoverOperator
from evolving_classifier.operators.MutationOperators import SomeStructMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from utility.Utility import *

if __name__ == '__main__':

    sm = Softmax()
    print(sm.compute(np.array(-1000)))
    print(sm.compute(np.array(1000)))

    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_y = one_hot_endode(train_y)
    test_y = one_hot_endode(test_y)

    train_X = [x.reshape(-1, 1) for x in train_X]
    test_X = [x.reshape(-1, 1) for x in test_X]

    hm = 1000
    train_X = train_X[:hm]
    train_y = train_y[:hm]
    test_X = test_X[:ceil(hm/3)]
    test_y = test_y[:ceil(hm/3)]

    ec = EvolvingClassifier()
    ec.hrange.layerCountMin = 0
    ec.hrange.layerCountMax = 2
    ec.co = SomeCrossoverOperator()
    ec.mo = SomeStructMutationOperator(ec.hrange)
    ec.so = TournamentSelection(2)
    ec.ff = CrossEffFitnessFunction()
    # TODO pousuwać zbędne argumenty z prepare
    ec.prepare(popSize=100, startPopSize=100, nn_data=(train_X, train_y, None, None), seed=1524)
    npoint = ec.run(50, 12)
    network = network_from_point(npoint, 1001)
    print(npoint.to_string())
    tests = network.test(test_X, test_y)
    print(tests)
    print(mean(tests[:3]))