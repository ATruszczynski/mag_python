import random

from ann_point.AnnPoint2 import AnnPoint2
from neural_network.ChaosNet import ChaosNet
from utility.AnnDataPoint import AnnDataPoint
from utility.Utility import choose_without_repetition, AnnPoint


class SelectionOperator:
    def __init__(self):
        pass

    def select(self, val_pop: [[ChaosNet, float]]) -> ChaosNet:
        pass


class TournamentSelection(SelectionOperator):
    def __init__(self, frac: float):
        super().__init__()
        self.frac = frac

    def select(self, val_pop: [[ChaosNet, AnnDataPoint]]) -> ChaosNet:
        count = max(round(len(val_pop) * self.frac), 2)

        chosen = choose_without_repetition(options=val_pop, count=count)
        chosen_sorted = sorted(chosen, key=lambda x: x[1].ff, reverse=True)
        return chosen_sorted[0][0].copy()

class RoulletteSelection(SelectionOperator):
    def __init__(self):
        super().__init__()

    def select(self, val_pop: [[ChaosNet, AnnDataPoint]]) -> ChaosNet:
        sum_fit = sum([val_pop[i][1].ff for i in range(len(val_pop))])
        roulette = [[val_pop[i][0], val_pop[i][1].ff/sum_fit] for i in range(len(val_pop))]
        for i in range(1, len(roulette)):
            roulette[i][1] += roulette[i - 1][1]
        rr = random.random()
        for i in range(len(roulette)):
            if rr < roulette[i][1]:
                return roulette[i][0].copy()