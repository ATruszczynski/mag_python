from typing import Any

from ann_point.HyperparameterRange import HyperparameterRange, assert_hranges_same
from evolving_classifier.operators.FinalCO1 import CrossoverOperator, LossFun
from evolving_classifier.operators.MutationOperators import MutationOperator
from evolving_classifier.operators.SelectionOperator import SelectionOperator, TournamentSelection
import numpy as np

from utility.Utility import copy_list_of_arrays


class TupleForTest:
    def __init__(self, name: str, rep: int, seed: int, popSize: int, data: [np.ndarray], iterations: int, hrange: HyperparameterRange,
                 ct: type, mt: type, st: [Any], fft: [Any], fct: type, reg: bool):
        self.name = name
        self.rep = rep
        self.seed = seed
        self.popSize = popSize
        self.data = []
        for i in range(len(data)):
            self.data.append(data[i].copy())
        self.iterations = iterations
        self.hrange = hrange.copy()
        self.ct = ct
        self.mt = mt
        self.st = st
        self.fft = fft
        self.fct = fct
        self.reg = reg

    def copy(self):
        x = copy_list_of_arrays(self.data[0])
        y = copy_list_of_arrays(self.data[1])
        X = copy_list_of_arrays(self.data[2])
        Y = copy_list_of_arrays(self.data[3])
        nData = (x, y, X, Y)
        return TupleForTest(name=self.name, rep=self.rep, seed=self.seed, popSize=self.popSize, data=nData, iterations=self.iterations,
                            hrange=self.hrange.copy(), ct=self.ct, mt=self.mt, st=self.st, fft=self.fft, fct=self.fct,
                            reg=self.reg)

def assert_tts_same(tt1: TupleForTest, tt2: TupleForTest):
    assert tt1.name == tt2.name
    assert tt1.rep == tt2.rep
    assert tt1.seed == tt2.seed
    assert tt1.popSize == tt2.popSize
    assert np.array_equal(tt1.data[0], tt2.data[0])
    assert np.array_equal(tt1.data[1], tt2.data[1])
    assert np.array_equal(tt1.data[2], tt2.data[2])
    assert np.array_equal(tt1.data[3], tt2.data[3])
    assert tt1.iterations == tt2.iterations
    assert_hranges_same(tt1.hrange, tt2.hrange)
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

