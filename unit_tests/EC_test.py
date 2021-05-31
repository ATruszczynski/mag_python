from evolving_classifier.EvolvingClassifier import *
from evolving_classifier.FitnessFunction import CrossEffFitnessFunction3
from evolving_classifier.operators.CrossoverOperator import MinimalDamageCrossoverOperator, WBCrossoverOperator
from evolving_classifier.operators.HillClimbOperator import HillClimbMutationOperator
from evolving_classifier.operators.MutationOperators import SomeStructMutationOperator, BiasedGaussianWBMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection


def test_determinism():
    ec = EvolvingClassifier()

    ec.sco = MinimalDamageCrossoverOperator()
    ec.co = WBCrossoverOperator()
    ec.smo = SomeStructMutationOperator(ec.hrange)
    ec.mo = BiasedGaussianWBMutationOperator(ec.hrange)
    ec.so = TournamentSelection(4)
    ec.ff = CrossEffFitnessFunction3()
    ec.hco = HillClimbMutationOperator(1, 5, ec.mo)

    count = 10
    size = 5
    x,y = generate_counting_problem(count, size)
    X,Y = generate_counting_problem(ceil(count), size)

    tests = []

    for i in range(5):
        ec.prepare(5, 5, (x, y, X, Y), 1001)
        net = ec.run(5, 5,  0.75, 0.01, 0.25, 1)
        tests.append(network_from_point(net, 1001).test(X, Y))

    for i in range(len(tests) - 1):
        for j in range(i + 1, len(tests)):
            t1 = tests[i]
            t2 = tests[j]

            assert t1[0] == t2[0]
            assert t1[1] == t2[1]
            assert t1[2] == t2[2]
            assert np.array_equal(t1[3], t2[3])
            assert t1[4] == t2[4]
