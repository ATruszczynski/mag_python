import random
import numpy as np

from TupleForTest import TupleForTest, assert_tts_same
from evolving_classifier.LsmFitnessCalculator import LsmFitnessCalculator
from evolving_classifier.FitnessFunction import CNFF, CNFF2, QuadDiff
from evolving_classifier.operators.LsmCrossoverOperator import LsmCrossoverOperator
from evolving_classifier.operators.LsmMutationOperator import LsmMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection, TournamentSelection95
from utility.Utility import generate_counting_problem, get_default_hrange_ga, assert_hranges_same


def test_tt_const():
    data = [np.zeros((1, 1)), np.zeros((2, 2)), np.zeros((3, 3)), np.zeros((4, 4))]
    hrange = get_default_hrange_ga()
    tt = TupleForTest(name="d", rep=1, seed=2, popSize=3, data=data, iterations=4, hrange=hrange,
                      ct=LsmCrossoverOperator, mt=LsmMutationOperator, st=[TournamentSelection95, 5], fft=[CNFF2, QuadDiff],
                      fct=LsmFitnessCalculator, reg=True)

    assert tt.name == "d"
    assert tt.rep == 1
    assert tt.seed == 2
    assert tt.popSize == 3
    assert np.array_equal(data[0], np.zeros((1, 1)))
    assert np.array_equal(data[1], np.zeros((2, 2)))
    assert np.array_equal(data[2], np.zeros((3, 3)))
    assert np.array_equal(data[3], np.zeros((4, 4)))
    assert tt.iterations == 4
    assert_hranges_same(tt.hrange, hrange)
    assert tt.ct == LsmCrossoverOperator
    assert tt.mt == LsmMutationOperator
    assert len(tt.st) == 2
    assert tt.st[0] == TournamentSelection95
    assert tt.st[1] == 5
    assert len(tt.fft) == 2
    assert tt.fft[0] == CNFF2
    assert tt.fft[1] == QuadDiff
    assert tt.fct == LsmFitnessCalculator
    assert tt.reg

    hrange.max_it = 222
    assert tt.hrange.max_it == 10

    data[2][-1, -1] = 222
    assert tt.data[2][-1, -1] == 0


def test_tt_copy():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    x, y = generate_counting_problem(10, 5)
    X, Y = generate_counting_problem(20, 5)
    hrange = get_default_hrange_ga()

    tt1 = TupleForTest(name="iris_01", rep=1, seed=random.randint(0, 10**6), popSize=2,
                       data=[x, y, X, Y], iterations=3, hrange=hrange,
                       ct=LsmCrossoverOperator, mt=LsmMutationOperator, st=[TournamentSelection, 4],
                       fft=[CNFF], fct=LsmFitnessCalculator, reg=False)

    tt2 = tt1.copy()

    assert_tts_same(tt1, tt2)

    tt1.hrange.max_swap = 222
    assert tt2.hrange.max_swap == -2

    tt1.data[3][2][-2, 0] = 333
    assert tt2.data[3][2][-2, 0] == 0

