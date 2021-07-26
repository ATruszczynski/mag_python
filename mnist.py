# from statistics import mean
#
#
# from evolving_classifier.EvolvingClassifier import EvolvingClassifier, network_from_point, OnlyFitnessCalculator
# from evolving_classifier.FitnessFunction import *
# from evolving_classifier.operators.CrossoverOperator import *
# from evolving_classifier.operators.MutationOperators import *
# from evolving_classifier.operators.SelectionOperator import TournamentSelection
# from utility.Utility import *
#
# if __name__ == '__main__':
#     #TODO podział na zbiór uczący EC, zbiór uczący NN i zbiór testowy?
#     from keras.datasets import mnist
#     (train_X, train_y), (test_X, test_y) = mnist.load_data()
#
#     train_y = one_hot_endode(train_y)
#     test_y = one_hot_endode(test_y)
#
#     train_X = [x.reshape(-1, 1) for x in train_X]
#     test_X = [x.reshape(-1, 1) for x in test_X]
#
#     hm = 2000
#     train_X = train_X[:hm]
#     train_y = train_y[:hm]
#     test_X = test_X[:ceil(hm/3)]
#     test_y = test_y[:ceil(hm/3)]
#
#     ec = EvolvingClassifier()
#     ec.hrange.hiddenLayerCountMin = 0
#     ec.hrange.hiddenLayerCountMax = 2
#     ec.hrange.neuronCountMax = 100
#     ec.co = SimpleCrossoverOperator()
#     ec.mo = SimpleAndStructuralCNMutation(ec.hrange)
#     ec.so = TournamentSelection(4)
#     ec.ff = CNFF()
#     ec.fc = OnlyFitnessCalculator([1, 0.6, 0.4, 0.25, 0.15, 0.1])
#     ec.prepare(popSize=50, startPopSize=50, nn_data=(train_X, train_y), seed=1524)
#     npoint = ec.run(iterations=20, pm=0.05, pc=0.8, power=12)
#     network = network_from_point(npoint, 1001)
#     network.train(train_X, train_y, 30)
#     print(npoint.to_string())
#     tests = network.test(test_X, test_y)
#     print(tests)
#     print(mean(tests[:3]))