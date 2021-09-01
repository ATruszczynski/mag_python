import random
from math import ceil

import numpy as np
from sklearn import datasets

from TupleForTest import TupleForTest
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO1 import FinalCO1
from evolving_classifier.operators.FinalCO2 import FinalCO2
from evolving_classifier.operators.FinalCO3 import FinalCO3
from evolving_classifier.operators.FinalCO4 import FinalCO4
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.MutationOperatorsP import FinalMutationOperatorP
from evolving_classifier.operators.PuzzleCO2 import PuzzleCO2
from evolving_classifier.operators.SelectionOperator import TournamentSelection, TournamentSelection05
from suites.suite_utility import try_check_if_all_tests_computable, trash_can, directory_for_tests
from tester import run_tests
from utility.Utility import one_hot_endode, get_default_hrange_ga, \
    get_default_hrange_es7, get_default_hrange_nmo, get_default_hrange_nco
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

    train_c = 125
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
        hrange = get_default_hrange_es7()

        tests = []

        repetitions = 2
        population_size = 500
        iterations = 150
        starg = max(2, ceil(0.05 * population_size))
        power = 12
        seed = 10011001

        # seeds = []
        # for i in range(4):
        #     seeds.append(random.randint(0, 10**6))

        # tests.append(TupleForTest(name=f"iris_avmax_lv_2", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))
        hrange = get_default_hrange_es7()
        # tests.append(TupleForTest(name=f"iris_avmax_ff6", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"iris_avmax_ff6_10", rep=repetitions, seed=seed, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 10],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_avmax_so_50", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 50],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_avmax", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 2*starg],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_meff", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))





        try_check_if_all_tests_computable(tests, trash_can, power=power)
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