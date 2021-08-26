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
from utility.Utility import one_hot_endode, get_default_hrange_ga, get_default_hrange_es, \
    generate_counting_problem_unique, generate_counting_problem
import os


def get_data():
    countTo = 4
    howMany = 300
    x, y = generate_counting_problem_unique(countTo=countTo)
    X, Y = generate_counting_problem(howMany=howMany, countTo=countTo)

    return (x, y, X, Y)

def test_suite_for_count():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()

        tests = []

        repetitions = 5
        population_size = 200
        iterations = 200
        starg = ceil(0.02 * population_size)
        power = 12

        seeds=[]
        for i in range(6):
            seeds.append(1002)

        hrange = get_default_hrange_ga()
        # hrange.max_hidden = 30


        # tests.append(TupleForTest(name="sos1_count_co1_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="sos2_count_co1_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos1_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos2_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos3_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
                                  fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos1_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection, starg],
                                  fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos2_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
                                  fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        tests.append(TupleForTest(name="cmp_sos3_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
                                  fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="sos5_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="sos5_count_co1_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="sos5_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized2, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))

        # tests.append(TupleForTest(name="ga_count_co1_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co1_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co1_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co1_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # tests.append(TupleForTest(name="ga_count_co2_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co2_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co2_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co2_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # tests.append(TupleForTest(name="ga_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co3_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="ga_count_co3_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = get_default_hrange_es()
        # hrange.max_hidden = 30
        #
        # tests.append(TupleForTest(name="es_count_co1_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co1_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co1_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co1_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # tests.append(TupleForTest(name="es_count_co2_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co2_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co2_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co2_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO2, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # tests.append(TupleForTest(name="es_count_co3_ff1", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co3_ff4", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co3_ff5", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF5], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name="es_count_co3_ff6", rep=repetitions, seed=seeds[3], popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
        #                           fft=[CNFF6, QuadDiff], fct=CNFitnessCalculator, reg=False))


        try_check_if_all_tests_computable(tests, trash_can, power=power)
        run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}combi_tests", power=power)
        # run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)

test_suite_for_count()