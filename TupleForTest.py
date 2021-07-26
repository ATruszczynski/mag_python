from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.CrossoverOperator import CrossoverOperator, LossFun
from evolving_classifier.operators.MutationOperators import MutationOperator
from evolving_classifier.operators.SelectionOperator import SelectionOperator, TournamentSelection
import numpy as np


class TupleForTest:
    # def __init__(self, rep: int, seed: int, popSize: int, data: [np.ndarray], iterations: int, hrange: HyperparameterRange,
    #              ff, fc, ct, mt, st, *starg):
    def __init__(self, rep: int, seed: int, popSize: int, data: [np.ndarray], iterations: int, hrange: HyperparameterRange,
                 ct: type, mt: type, st: type, fft: type, fct: type, starg: float, fftarg: type, reg: bool):
        self.rep = rep
        self.seed = seed
        self.popSize = popSize
        self.iterations = iterations
        self.hrange = hrange
        self.ct = ct
        self.mt = mt
        self.st = st
        self.starg = starg
        self.fft = fft
        self.fct = fct
        self.fftarg = fftarg
        self.data = data
        self.reg = reg

