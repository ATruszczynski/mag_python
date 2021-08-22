from typing import Any

from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.FinalCO1 import CrossoverOperator, LossFun
from evolving_classifier.operators.MutationOperators import MutationOperator
from evolving_classifier.operators.SelectionOperator import SelectionOperator, TournamentSelection
import numpy as np

class TupleForTest:
    def __init__(self, name: str, rep: int, seed: int, popSize: int, data: [np.ndarray], iterations: int, hrange: HyperparameterRange,
                 ct: type, mt: type, st: [Any], fft: [Any], fct: type, reg: bool):
        self.name = name
        self.rep = rep
        self.seed = seed
        self.popSize = popSize
        self.iterations = iterations
        self.hrange = hrange
        self.ct = ct
        self.mt = mt
        self.st = st
        self.fft = fft
        self.fct = fct
        self.data = data
        self.reg = reg

