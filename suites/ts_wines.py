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
from evolving_classifier.operators.FinalCO4 import FinalCO4
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.MutationOperatorsP import FinalMutationOperatorP
from evolving_classifier.operators.PuzzleCO2 import PuzzleCO2
from evolving_classifier.operators.SelectionOperator import TournamentSelection, TournamentSelectionSized, \
    TournamentSelectionSized2, TournamentSelection06, RoulletteSelection
from suites.suite_utility import try_check_if_all_tests_computable, trash_can, directory_for_tests
from tester import run_tests
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es, get_default_hrange_es3, \
    translate_wines, divide_frame_into_columns, get_default_hrange_es4, get_default_hrange_es5, get_default_hrange_es6, \
    get_default_hrange_es7
import os
import pandas as pd


def get_data():
    data_frame = pd.read_csv(fr"..{os.path.sep}data_sets{os.path.sep}winequality-white.csv")

    cols_to_norm = [1, 4, 12]
    cols_to_norm = data_frame.columns[:-1]
    data_frame[cols_to_norm] = StandardScaler().fit_transform(data_frame[cols_to_norm])

    qualities = data_frame["quality"].unique()
    tt = 25
    frames = []
    for i in qualities:
        df = data_frame.loc[data_frame["quality"] == i].iloc[:tt, :]
        if df.shape[0] >= 5:
            frames.append(df)

    data_frame = pd.concat(frames,ignore_index=True)
    data_frame = data_frame.sample(frac=1)

    div = ceil(0.8 * data_frame.shape[0])

    train = data_frame.iloc[:div, :]
    print(len(train["quality"].unique()))
    test = data_frame.iloc[div:, :]

    x = divide_frame_into_columns(train.iloc[:, :-1].transpose())
    X = divide_frame_into_columns(test.iloc[:, :-1].transpose())

    Ys = one_hot_endode(data_frame.iloc[:, -1].tolist())
    # Ys = data_frame.iloc[:, [-1]].to_numpy()
    y = Ys[:div]
    Y = Ys[div:]


    return (x, y, X, Y)

def test_suite_for_wine():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()

        tests = []

        repetitions = 10
        population_size = 500
        iterations = 300
        # population_size = 250
        # iterations = 100
        starg = max(2, ceil(0.1 * population_size))
        power = 12
        seed = 1001

        seeds = []
        for i in range(5):
            seeds.append(random.randint(0, 10**6))
        hrange = get_default_hrange_es7()

        # tests.append(TupleForTest(name=f"wines10_co3", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"wines16_50_10_2_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"wines_nm_1_2", rep=2, seed=seeds[0], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"wines_nm_2_2", rep=2, seed=seeds[1], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"wines_nm_3_2", rep=2, seed=seeds[2], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"wines_nm_4_2", rep=2, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name=f"wines_nm_5_2", rep=2, seed=seeds[4], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
                                  fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"wines16_10", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"wines2_co1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"wines2_co3", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection06, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))


        # try_check_if_all_tests_computable(tests, trash_can, power=power)
        res = run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)


        net = res[0][0]

        # print(net.run(x[0]))
        # print(net.run(x[1]))
        # print(net.run(x[2]))




        restr = net.test(test_input=x, test_output=y)
        print(restr[0])
        print(m_efficiency(restr[0]))
        print(efficiency(restr[0]))
        res = net.test(test_input=X, test_output=Y)
        print(res[0])
        np.set_printoptions(suppress=True)
        print(m_efficiency(res[0]))
        print(efficiency(res[0]))
        if len(res) == 2:
            print(res[1])
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_wine()