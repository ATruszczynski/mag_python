from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO1 import *
from evolving_classifier.operators.FinalCO2 import FinalCO2
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from tester import run_tests
from TupleForTest import TupleForTest
from utility.TestingUtility import compare_chaos_networks
from utility.Utility import *


def test_tester_same_as_ec_ind():
    if __name__ == '__main__':
        hrange = get_default_hrange()
        io = generate_counting_problem(100, 5)

        seed = 22223333
        random.seed(seed)
        np.random.seed(seed)

        count_tr = 500
        count_test = 500
        size = 7

        x,y = generate_counting_problem(count_tr, size)
        X,Y = generate_counting_problem(ceil(count_test), size)

        popSize = 10
        iterations = 10
        seed = 1001
        power=1
        how_many = 10

        hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                     mut_radius=(-2, 0), sqr_mut_prob=(-2, 0), lin_mut_prob=(-2, 0), p_mutation_prob=(-2, 0), c_prob=(-2, 0),
                                     dstr_mut_prob=(-2, 0))

        test = TupleForTest(name="test_desu", rep=how_many, seed=seed, popSize=popSize, data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                            ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, 2],
                            fft=[CNFF], fct=CNFitnessCalculator, reg=False)

        results = run_tests([test], power=power)[0]

        random.seed(seed)
        seeds = []
        for i in range(how_many):
            seeds.append(random.randint(0, 10**6))

        results2 = []
        for i in range(how_many):
            ec = EvolvingClassifier()
            ec.prepare(popSize=popSize, nn_data=[x, y], seed=seeds[i], hrange=hrange)
            results2.append(ec.run(iterations=iterations, power=power))

        for i in range(how_many):
            compare_chaos_networks(results[i], results2[i])


def test_tester_determinism():
    if __name__ == '__main__':
        # hrange = get_default_hrange()
        # io = generate_counting_problem(100, 5)

        seed = 22223333
        random.seed(seed)
        np.random.seed(seed)

        count_tr = 10
        count_test = 10
        size = 4

        x,y = generate_counting_problem(count_tr, size)
        X,Y = generate_counting_problem(ceil(count_test), size)

        popSize = 10
        iterations = 10
        seed = 1001
        power=1
        rep = 5
        ut_rep = 10

        hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                     mut_radius=(-2, 0), sqr_mut_prob=(-2, 0), lin_mut_prob=(-2, 0), p_mutation_prob=(-2, 0), c_prob=(-2, 0),
                                     dstr_mut_prob=(-2, 0))

        test = TupleForTest(name="test_desu", rep=rep, seed=seed, popSize=popSize, data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                            ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, 2],
                            fft=[CNFF], fct=CNFitnessCalculator, reg=False)

        hrange = HyperparameterRange((-2, 1), (-1, 2), (2, 6), (4, 20), [Poly2(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                     mut_radius=(-2, 0), sqr_mut_prob=(-3, 0), lin_mut_prob=(-2, 0), p_mutation_prob=(-2, 0), c_prob=(-2, 0),
                                     dstr_mut_prob=(-3, 0))

        test2 = TupleForTest(name="test_desu2", rep=2*rep, seed=2*seed, popSize=2*popSize, data=[x, y, X, Y], iterations=2*iterations, hrange=hrange,
                            ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelection, 3],
                            fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False)

        resultsss = []
        for i in range(ut_rep):
            resultsss.append(run_tests([test, test2], power=power))

        for i in range(len(resultsss) - 1):
            resultss1 = resultsss[i]
            resultss2 = resultsss[i + 1]

            assert len(resultss1) == 2
            assert len(resultss2) == 2

            for j in range(len(resultss1)):
                results1 = resultss1[j]
                results2 = resultss2[j]

                assert len(results1) == (j + 1) * rep
                assert len(results2) == (j + 1) * rep

                for k in range(len(results1)):
                    net1 = results1[k]
                    net2 = results2[k]
                    compare_chaos_networks(net1, net2)


# test_tester_same_as_ec_ind()
test_tester_determinism()




