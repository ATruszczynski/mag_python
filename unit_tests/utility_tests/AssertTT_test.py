import numpy as np

from TupleForTest import TupleForTest, assert_tts_same
from ann_point.Functions import QuadDiff, CrossEntropy
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import CNFF2, CNFF, CNFF4
from evolving_classifier.operators.Rejects.FinalCO1 import FinalCO1
from evolving_classifier.operators.Rejects.FinalCO2 import FinalCO2
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from utility.MockFC import MockFC
from utility.MockMutOp import MockMO
from utility.Utility import get_testing_hrange


def get_testing_tt():
    return TupleForTest(
        name="desu",
        rep=1,
        seed=2,
        popSize=3,
        data=[np.zeros((4, 4)), np.zeros((5, 5)), np.zeros((6, 6)), np.zeros((7, 7))],
        iterations=8,
        hrange=get_testing_hrange(),
        ct=FinalCO1,
        mt=FinalMutationOperator,
        st=[TournamentSelection, 9],
        fft=[CNFF2, QuadDiff],
        fct=CNFitnessCalculator,
        reg=False
    )


def test_same():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)


def test_different_name():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.name = "ddd"

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_rep():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.rep = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_seed():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.seed = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_popSize():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.popSize = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_data_0():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.data[0][-2, -2] = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_data_1():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.data[1][-3, -3] = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_data_2():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.data[2][-4, -4] = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_data_3():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.data[3][-5, -5] = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_iterations():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.iterations = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_hrange():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.hrange.min_it = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_ct():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.ct = FinalCO2

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_mt():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.mt = MockMO

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_st_wrong_size():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.st = [TournamentSelection]

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False

#
# def test_different_st_wrong_func():
#     tt1 = get_testing_tt()
#     tt2 = get_testing_tt()
#
#     assert_tts_same(tt1, tt2)
#
#     tt2.st = [TournamentSelectionSized, 9]
#
#     try:
#         assert_tts_same(tt1, tt2)
#     except AssertionError:
#         assert True
#     else:
#         assert False


def test_different_st_wrong_arg():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.st[1] = -100

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_fft_wrong_size():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.fft = [CNFF]

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_fft_wrong_ff():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.fft = [CNFF4, QuadDiff]

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_fft_wrong_func():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.fft[1] = CrossEntropy

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_ffc():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.fct = MockFC

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False


def test_different_reg():
    tt1 = get_testing_tt()
    tt2 = get_testing_tt()

    assert_tts_same(tt1, tt2)

    tt2.reg = True

    try:
        assert_tts_same(tt1, tt2)
    except AssertionError:
        assert True
    else:
        assert False



