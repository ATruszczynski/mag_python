import random

from ann_point.AnnPoint2 import AnnPoint2
from utility.AnnDataPoint import AnnDataPoint
from utility.Utility import choose_without_repetition, AnnPoint


class SelectionOperator:
    def __init__(self):
        pass

    def select(self, val_pop: [[AnnPoint2, float]]) -> AnnPoint2:
        pass


class TournamentSelection(SelectionOperator):
    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def select(self, val_pop: [[AnnPoint, AnnDataPoint]]) -> AnnPoint:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        chosen_sorted = sorted(chosen, key=lambda x: x[1].ff, reverse=True)
        return chosen_sorted[0][0].copy()

class RoulletteSelection(SelectionOperator):
    def __init__(self):
        super().__init__()

    def select(self, val_pop: [[AnnPoint, AnnDataPoint]]) -> AnnPoint:
        sum_fit = sum([val_pop[i][1].ff for i in range(len(val_pop))])
        roulette = [[val_pop[i][0], val_pop[i][1].ff/sum_fit] for i in range(len(val_pop))]
        for i in range(1, len(roulette)):
            roulette[i][1] += roulette[i - 1][1]
        rr = random.random()
        for i in range(len(roulette)):
            if rr < roulette[i][1]:
                return roulette[i][0].copy()