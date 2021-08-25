import random
import numpy as np
from sklearn import datasets

from TupleForTest import TupleForTest
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO1 import FinalCO1
from evolving_classifier.operators.FinalCO2 import FinalCO2
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection, TournamentSelectionSized
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

        # TODO - SS - delete
        hrange.max_hidden = 50

        tests = []

        repetitions = 3
        population_size = 100
        iterations = 100
        starg = 3

        tests.append(TupleForTest(name="iris_01", rep=repetitions, seed=-1, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_02", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_03", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_04", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))

        tests.append(TupleForTest(name="iris_01_sized", rep=repetitions, seed=-1, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_02_sized", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_03_sized", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="iris_04_sized", rep=repetitions, seed=-1, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))

        for i in range(len(tests)):
            tests[i].seed = random.randint(0, 10**6)

        try_check_if_all_tests_computable(tests, trash_can, 12)
        run_tests(tts=tests, directory_for_tests=directory_for_tests, power=12)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=12)

test_suite_for_iris()