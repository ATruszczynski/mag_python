import random
from math import ceil

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
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es, \
    generate_counting_problem_unique, generate_counting_problem
import os


def get_data():
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_y = one_hot_endode(train_y)
    test_y = one_hot_endode(test_y)

    train_X = [x.reshape(-1, 1) for x in train_X]
    test_X = [x.reshape(-1, 1) for x in test_X]

    hm = 2000
    train_X = train_X[:hm]
    train_y = train_y[:hm]
    test_X = test_X[:ceil(hm/3)]
    test_y = test_y[:ceil(hm/3)]

    return (train_X, train_y, test_X, test_y)

def test_suite_for_mnist():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()
        hrange = get_default_hrange_ga()

        # TODO - SS - delete
        hrange.max_hidden = 50

        tests = []

        repetitions = 1
        population_size = 100
        iterations = 100
        starg = 3

        power = 12

        tests.append(TupleForTest(name="mnist_01", rep=repetitions, seed=-1, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))


        for i in range(len(tests)):
            tests[i].seed = random.randint(0, 10**6)

        try_check_if_all_tests_computable(tests, trash_can, power)
        run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=12)

test_suite_for_mnist()