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
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es, get_default_hrange_es3, \
    translate_wines, divide_frame_into_columns
import os
import pandas as pd


def get_data():
    data_frame = pd.read_csv(fr"..{os.path.sep}data_sets{os.path.sep}winequality-white.csv")
    data_frame = data_frame.sample(frac=1)

    cols_to_norm = [1, 4, 12]
    cols_to_norm = data_frame.columns[:-1]
    data_frame[cols_to_norm] = StandardScaler().fit_transform(data_frame[cols_to_norm])

    div = 400

    train = data_frame.iloc[:div, :]
    test = data_frame.iloc[div:, :]

    x = divide_frame_into_columns(train.iloc[:, :-1].transpose())
    X = divide_frame_into_columns(test.iloc[:, :-1].transpose())

    Ys = one_hot_endode(data_frame.iloc[:, -1].tolist())
    y = Ys[:div]
    Y = Ys[div:]

    return (x, y, X, Y)

def test_suite_for_wine():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()
        hrange = get_default_hrange_ga()
        hrange = get_default_hrange_es3()

        tests = []

        repetitions = 1
        population_size = 200
        iterations = 300
        starg = ceil(0.02 * population_size)
        power = 12
        seed = 1001

        seeds = []
        for i in range(4):
            seeds.append(random.randint(0, 10**6))


        tests.append(TupleForTest(name=f"wines", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, 2],
                                  fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))


        # try_check_if_all_tests_computable(tests, trash_can, power=power)
        res = run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)


        net = res[0][0]
        restr = net.test(test_input=x, test_output=y)
        print(restr[0])
        print(m_efficiency(restr[0]))
        res = net.test(test_input=X, test_output=Y)
        print(res[0])
        np.set_printoptions(suppress=True)
        print(m_efficiency(res[0]))
        if len(res) == 2:
            print(res[1])
        print(5 * res[0][1, 0] + res[0][0, 1])
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_wine()