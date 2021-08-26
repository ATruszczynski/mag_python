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
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection, TournamentSelectionSized, \
    TournamentSelectionSized2
from suites.suite_utility import try_check_if_all_tests_computable, trash_can, directory_for_tests
from tester import run_tests
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es
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
        hrange = get_default_hrange_ga()

        tests = []

        repetitions = 5
        population_size = 100
        iterations = 100
        starg = ceil(0.02 * population_size)
        power = 12

        # TODO - SS - to delete
        hrange.max_hidden = 50


        seeds = []
        for i in range(4):
            seeds.append(random.randint(0, 10**6))

        # finals
        # tests.append(TupleForTest(name=f"iris_co2_ff1_sos_2", rep=repetitions, seed=seeds[0], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_co2_ff4_qd_sos_2", rep=repetitions, seed=seeds[1], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_co2_ff5_sos_2", rep=repetitions, seed=seeds[2], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"iris_co2_ff6_qd_sos_2", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))

        # tests.append(TupleForTest(name=f"iris_sos_001_ff", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"iris_sos_001_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
                                  fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))



        try_check_if_all_tests_computable(tests, trash_can, power=power)
        run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_iris()