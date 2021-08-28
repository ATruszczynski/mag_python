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
    generate_counting_problem_unique, generate_counting_problem, translate_german, divide_frame_into_columns, \
    get_default_hrange_ga2, get_default_hrange_es2, get_default_hrange_es3
import os
import pandas as pd

def get_data():
    div = 800

    data_frame = pd.read_csv(fr"..{os.path.sep}data_sets{os.path.sep}translated_german.csv", header=None)
    data_frame = data_frame.astype('float')

    cols_to_norm = [1, 4, 12]
    cols_to_norm = list(range(18))
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
        population_size = 100
        iterations = 300
        starg = max(ceil(0.05 * population_size), 2)
        starg = 2
        power = 12

        seeds=[]
        for i in range(6):
            seeds.append(10011002)

        hrange = get_default_hrange_es3()

        # tests.append(TupleForTest(name=f"german_ff7", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_ff8", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF8, QuadDiff], fct=CNFitnessCalculator, reg=False))


        # tests.append(TupleForTest(name=f"german_3_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, 10],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_3_25", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, 25],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        #
        # tests.append(TupleForTest(name=f"german_2_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 10],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"german_10_2", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 2],
                                  fft=[CNFF9], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_8_5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 5],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_8_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 10],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_7_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 10],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_7_25", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, 25],
        #                           fft=[CNFF7], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"german_ff5_ga43es", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF8, QuadDiff], fct=CNFitnessCalculator, reg=False))


        # try_check_if_all_tests_computable(tests, trash_can, power=power)
        res = run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        net = res[0][0]
        restr = net.test(test_input=x, test_output=y)
        print(restr[0])
        print(m_efficiency(restr[0]))
        res = net.test(test_input=X, test_output=Y)
        print(res[0])
        print(m_efficiency(res[0]))
        if len(res) == 2:
            print(res[1])
        print(5 * res[0][1, 0] + res[0][0, 1])
        # print(net.inp[net.hidden_start_index:])
        # print(net.weights[0:net.hidden_end_index, net.hidden_start_index:])

        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_german()