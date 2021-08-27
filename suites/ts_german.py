import random
from math import ceil

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

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
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es, \
    generate_counting_problem_unique, generate_counting_problem, translate_german, divide_frame_into_columns
import os
import pandas as pd

def get_data():
    div = 200

    data_frame = pd.read_csv(fr"..{os.path.sep}data_sets{os.path.sep}translated_german.csv", header=None)

    cols_to_norm = [1, 4, 12]
    data_frame[cols_to_norm] = StandardScaler().fit_transform(data_frame[cols_to_norm])

    train = data_frame.iloc[:div, :]
    test = data_frame.iloc[div:, :]

    x = divide_frame_into_columns(train.iloc[:, :-1].transpose())
    y = one_hot_endode(train.iloc[:, -1].tolist())
    X = divide_frame_into_columns(test.iloc[:, :-1].transpose())
    Y = one_hot_endode(test.iloc[:, -1].tolist())

    return (x, y, X, Y)

def test_suite_for_german():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()

        tests = []

        repetitions = 1
        population_size = 500
        iterations = 500
        starg = max(ceil(0.02 * population_size), 2)
        power = 12

        seeds=[]
        for i in range(6):
            seeds.append(10011001)

        hrange = get_default_hrange_ga()
        hrange.max_hidden = 75


        # tests.append(TupleForTest(name=f"german_ff7", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_ff8", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF8, QuadDiff], fct=CNFitnessCalculator, reg=False))


        hrange = get_default_hrange_es()
        tests.append(TupleForTest(name=f"german_ff5_ga32", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
                                  fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_ff5_ga43es", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF8, MeanDiff], fct=CNFitnessCalculator, reg=False))


        try_check_if_all_tests_computable(tests, trash_can, power=power)
        run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_german()