import random
from math import ceil

from sklearn import datasets

from TupleForTest import TupleForTest
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO3 import FinalCO3
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection05
from suites.suite_utility import try_check_if_all_tests_computable, trash_can, directory_for_tests
from tester import run_tests
from utility.Utility import one_hot_endode, get_doc_hrange_qd, get_doc_hrange_eff
import os

def get_data():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x = [x.reshape((4, 1)) for x in x]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    xx = [x[i] for i in perm]
    yy = [y[i] for i in perm]

    train_c = 105
    x = [xx[i] for i in range(train_c)]
    y = [yy[i] for i in range(train_c)]
    X = [xx[i] for i in range(train_c, 150)]
    Y = [yy[i] for i in range(train_c, 150)]

    return (x, y, X, Y)





def test_suite_for_iris():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()

        tests = []

        repetitions = 5
        population_size = 1000
        iterations = 200
        starg = 4
        power = 12
        seed = 10011001

        # seeds = []
        # for i in range(4):
        #     seeds.append(random.randint(0, 10**6))

        hrange = get_doc_hrange_eff()
        tests.append(TupleForTest(name=f"iris_500_200_meff", rep=repetitions, seed=seed, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
                                  fft=[MEFF], fct=CNFitnessCalculator, reg=False))

        hrange = get_doc_hrange_qd()
        tests.append(TupleForTest(name=f"iris_500_200_avmax", rep=repetitions, seed=seed, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
                                  fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))

        hrange = get_doc_hrange_qd()
        tests.append(TupleForTest(name=f"iris_500_200_mixmeff", rep=repetitions, seed=seed, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
                                  fft=[MIXFF, QuadDiff], fct=CNFitnessCalculator, reg=False))


        # try_check_if_all_tests_computable(tests, trash_can, power=power)
        net = run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)[0][0]
        # net = run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)[0][0]
        # for jjj in range(len(x)):
        #     print(f"{net.run(x[jjj]).T} - {y[jjj].T}")
        restr = net.test(test_input=x, test_output=y)
        print(restr[0])
        print(m_efficiency(restr[0]))
        res = net.test(test_input=X, test_output=Y)
        print(res[0])
        np.set_printoptions(suppress=True)
        print(m_efficiency(res[0]))
        if len(res) == 2:
            print(res[1])

test_suite_for_iris()