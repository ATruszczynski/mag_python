from TupleForTest import TupleForTest
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import CNFF
from evolving_classifier.operators.Rejects.FinalCO1 import FinalCO1
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from suites.suite_utility import try_check_if_all_tests_computable
import os

from utility.Utility import get_default_hrange_ga, generate_counting_problem

trash_can = f"..{os.path.sep}..{os.path.sep}algo_tests{os.path.sep}to_delete{os.path.sep}uts"


def test_check_different():
    tests = []
    repetitions = 1
    seed = 2
    population_size = 3
    x, y = generate_counting_problem(10, 5)
    X, Y = generate_counting_problem(10, 5)
    data = [x, y, X, Y]
    iterations = 8
    hrange = get_default_hrange_ga()
    ct=FinalCO1
    mt=FinalMutationOperator
    st=[TournamentSelection, 4]
    fft=[CNFF]
    fct=CNFitnessCalculator
    reg=False

    tests.append(TupleForTest(name="iris_01", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_02", rep=repetitions, seed=seed+1, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_03", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations+1, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))

    try_check_if_all_tests_computable(tests=tests, directory=trash_can, power=1)


def test_check_same_names():
    tests = []
    repetitions = 1
    seed = 2
    population_size = 3
    x, y = generate_counting_problem(10, 5)
    X, Y = generate_counting_problem(10, 5)
    data = [x, y, X, Y]
    iterations = 8
    hrange = get_default_hrange_ga()
    ct=FinalCO1
    mt=FinalMutationOperator
    st=[TournamentSelection, 4]
    fft=[CNFF]
    fct=CNFitnessCalculator
    reg=False

    tests.append(TupleForTest(name="iris_01", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_02", rep=repetitions, seed=seed+1, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_01", rep=repetitions, seed=seed+2, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))

    try:
        try_check_if_all_tests_computable(tests=tests, directory=trash_can, power=1)
    except AssertionError:
        assert True
    else:
        assert False


def test_check_same_params_diff_names():
    tests = []
    repetitions = 1
    seed = 2
    population_size = 3
    x, y = generate_counting_problem(10, 5)
    X, Y = generate_counting_problem(10, 5)
    data = [x, y, X, Y]
    iterations = 8
    hrange = get_default_hrange_ga()
    ct=FinalCO1
    mt=FinalMutationOperator
    st=[TournamentSelection, 4]
    fft=[CNFF]
    fct=CNFitnessCalculator
    reg=False

    tests.append(TupleForTest(name="iris_01", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_02", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))
    tests.append(TupleForTest(name="iris_03", rep=repetitions, seed=seed, popSize=population_size,
                              data=data, iterations=iterations, hrange=hrange,
                              ct=ct, mt=mt, st=st, fft=fft, fct=fct, reg=reg))

    try:
        try_check_if_all_tests_computable(tests=tests, directory=trash_can, power=1)
    except AssertionError:
        assert True
    else:
        assert False


# test_check_different()
# test_check_same_names()
# test_check_same_params_diff_names()