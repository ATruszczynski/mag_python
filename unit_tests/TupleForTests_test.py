import numpy as np
import random

from TupleForTest import TupleForTest, assert_tts_same
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import CNFF
from evolving_classifier.operators.FinalCO1 import FinalCO1
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from utility.Utility import generate_counting_problem, get_default_hrange


def test_tt_copy():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)
    x, y = generate_counting_problem(10, 5)
    X, Y = generate_counting_problem(20, 5)
    hrange = get_default_hrange()

    tt1 = TupleForTest(name="iris_01", rep=1, seed=random.randint(0, 10**6), popSize=2,
                              data=[x, y, X, Y], iterations=3, hrange=hrange,
                              ct=FinalCO1, mt=FinalMutationOperator, st=[TournamentSelection, 4],
                              fft=[CNFF], fct=CNFitnessCalculator, reg=False)

    tt2 = tt1.copy()

    assert tt1.name == tt2.name
    assert tt1.rep == tt2.rep
    assert tt1.seed == tt2.seed
    assert tt1.popSize == tt2.popSize
    assert np.array_equal(tt1.data[0], tt2.data[0])
    assert np.array_equal(tt1.data[1], tt2.data[1])
    assert np.array_equal(tt1.data[2], tt2.data[2])
    assert np.array_equal(tt1.data[3], tt2.data[3])
    assert tt1.iterations == tt2.iterations
    assert tt1.hrange == tt2.hrange
    assert tt1.ct == tt2.ct
    assert tt1.mt == tt2.mt
    assert len(tt1.st) == len(tt2.st)
    for i in range(len(tt1.st)):
        assert tt1.st[i] == tt2.st[i]
    assert len(tt1.fft) == len(tt2.fft)
    for i in range(len(tt1.fft)):
        assert tt1.fft[i] == tt2.fft[i]
    assert tt1.fct == tt2.fct
    assert tt1.reg == tt2.reg

test_tt_copy()
