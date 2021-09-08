import random
from math import ceil

from sklearn.preprocessing import StandardScaler

from TupleForTest import TupleForTest
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.Rejects.FinalCO1 import HyperparameterRange
from evolving_classifier.operators.FinalCO3 import FinalCO3
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection05
from suites.suite_utility import try_check_if_all_tests_computable, trash_can, directory_for_tests
from tester import run_tests
from utility.Utility import one_hot_endode, \
    divide_frame_into_columns
import os
import pandas as pd


def get_data():
    data_frame = pd.read_csv(fr"..{os.path.sep}data_sets{os.path.sep}winequality-white.csv")
    # data_frame = pd.read_csv(fr"data_sets{os.path.sep}winequality-white.csv")

    cols_to_norm = [1, 4, 12]
    cols_to_norm = data_frame.columns[:-1]
    data_frame[cols_to_norm] = StandardScaler().fit_transform(data_frame[cols_to_norm])
    data_frame = data_frame.sample(frac=1)

    qualities = data_frame["quality"].unique()
    tt = 200
    frac = 0.6

    train_frames = []
    test_frames = []
    for i in qualities:
        df = data_frame.loc[data_frame["quality"] == i].iloc[:tt, :]
        div = ceil(frac * df.shape[0])
        train_frames.append(df.iloc[:div, :])
        test_frames.append(df.iloc[div:, :])

    train_frame = pd.concat(train_frames, ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)

    print(len(train_frame["quality"].unique()))

    x = divide_frame_into_columns(train_frame.iloc[:, :-1].transpose())
    X = divide_frame_into_columns(test_frame.iloc[:, :-1].transpose())

    y = one_hot_endode(train_frame.iloc[:, -1].tolist())
    Y = one_hot_endode(test_frame.iloc[:, -1].tolist())

    return (x, y, X, Y)

# def get_wines_hrange_soc():
#     d = 0.000
#     dd = (-d, d)
#     ddd = (-d*10, d*10)
#
#     hrange = HyperparameterRange(init_wei=dd, init_bia=ddd, it=(1, 5), hidden_count=(1, 30),
#                                  actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()],
#                                  mut_radius=(-4, -1),
#                                  c_prob=(log10(0.5), log10(0.5)),
#                                  multi=(-2, 2),
#                                  p_prob=(log10(0.01), log10(1)),
#                                  p_rad=(log10(0.01), log10(0.1)))
#
#     return hrange


def get_wines_hrange_eff():
    d = 0.000
    dd = (-d, d)
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=dd, init_bia=ddd, it=(1, 5), hidden_count=(1, 10),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-4, -0),
                                 multi=(-2, 2),
                                 c_prob=(log10(0.5), log10(0.5)),
                                 swap=(log10(0.1), log10(0.1)),
                                 p_prob=(log10(0.1), log10(0.1)),
                                 p_rad=(log10(0.1), log10(0.1)),
                                 aggrFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Softmax(), Identity(), Poly2(), Poly3()])

    return hrange


def get_wines_hrange_avmax():
    d = 0.0001
    dd = (-d, d)
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=dd, init_bia=ddd, it=(1, 5), hidden_count=(1, 100),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-4, -0),
                                 multi=(-2, 2),
                                 c_prob=(log10(0.5), log10(0.5)),
                                 swap=(log10(0.1), log10(0.1)),
                                 p_prob=(log10(0.1), log10(0.1)),
                                 p_rad=(log10(0.1), log10(0.1)),
                                 aggrFuns=[Identity()])

    return hrange

def get_wines_hrange_mixff():
    d = 0.0001
    dd = (-d, d)
    ddd = (-d, d)

    hrange = HyperparameterRange(init_wei=dd, init_bia=ddd, it=(1, 5), hidden_count=(20, 20),
                                 actFuns=[ReLu(), LReLu(), GaussAct(), SincAct(), TanH(), Sigmoid(), Identity(), Poly2(), Poly3()],
                                 mut_radius=(-4, -0),
                                 multi=(-2, 2),
                                 c_prob=(log10(0.5), log10(0.5)),
                                 swap=(log10(0.1), log10(0.1)),
                                 p_prob=(log10(0.1), log10(0.1)),
                                 p_rad=(log10(0.1), log10(0.1)),
                                 aggrFuns=[Identity()])
    return hrange


def test_suite_for_wine():
    if __name__ == '__main__':
        seed = 1001
        random.seed(seed)
        np.random.seed(seed)

        x, y, X, Y = get_data()

        tests = []

        repetitions = 5
        population_size = 1000
        iterations = 200
        starg = max(2, ceil(0.01 * population_size))
        power = 12
        seed = 1001

        seeds = []
        for i in range(5):
            seeds.append(random.randint(0, 10**6))

        # hrange = get_wines_hrange_eff()
        # tests.append(TupleForTest(name=f"wines_10_EFF", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = get_wines_hrange_avmax()
        # tests.append(TupleForTest(name=f"wwwwines_AVMAX", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))

        hrange = get_wines_hrange_mixff()
        tests.append(TupleForTest(name=f"wines_20_20_MIXFF", rep=repetitions, seed=seed, popSize=population_size,
                                  data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                                  ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
                                  fft=[MIXFF, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = get_wines_hrange_qd()
        # hrange.min_mut_radius = -2.5
        # hrange.max_mut_radius = -2.5
        # tests.append(TupleForTest(name=f"QD_7", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = get_wines_hrange_qd()
        # hrange.min_mut_radius = -3
        # hrange.max_mut_radius = -3
        # tests.append(TupleForTest(name=f"QD_8", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, starg],
        #                           fft=[AVMAX, QuadDiff], fct=CNFitnessCalculator, reg=False))


        # tests.append(TupleForTest(name=f"AT_wines_dco_10", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 10],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"AT_wines_dco_20", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 20],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))

        # hrange = hrange.copy()
        # hrange.min_hidden = 80
        # hrange.max_hidden = 100
        # tests.append(TupleForTest(name=f"AT_wines_dco_size_5", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))

        # hrange = hrange.copy()
        # hrange.min_hidden = 30
        # hrange.max_hidden = 50
        # tests.append(TupleForTest(name=f"AT_wines_dco_size_4", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = hrange.copy()
        # hrange.min_hidden = 0
        # hrange.max_hidden = 25
        # tests.append(TupleForTest(name=f"AT_wines_dco_size_3", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = hrange.copy()
        # hrange.min_hidden = 0
        # hrange.max_hidden = 10
        # tests.append(TupleForTest(name=f"AT_wines_dco_size_2", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        #
        # hrange = hrange.copy()
        # hrange.min_hidden = 0
        # hrange.max_hidden = 5
        # tests.append(TupleForTest(name=f"AT_wines_dco_size_1", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"AT_wines_dco_prad0", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 4],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"AT_wines_dco_10", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 10],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))
        # tests.append(TupleForTest(name=f"AT_wines_dco_20", rep=repetitions, seed=seed, popSize=population_size,
        #                           data=[x, y, X, Y], iterations=iterations, hrange=hrange,
        #                           ct=FinalCO3, mt=FinalMutationOperator, st=[TournamentSelection05, 20],
        #                           fft=[MEFF], fct=CNFitnessCalculator, reg=False))




        try_check_if_all_tests_computable(tests, trash_can, power=power)
        # res = run_tests(tts=tests, directory_for_tests=directory_for_tests, power=power)
        res = run_tests(tts=tests, directory_for_tests=f"..{os.path.sep}final_tests", power=power)
        net = res[0][0]
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


test_suite_for_wine()