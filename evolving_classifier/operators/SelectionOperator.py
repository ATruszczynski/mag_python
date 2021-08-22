import random

from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint
from utility.Utility import choose_without_repetition


class SelectionOperator:
    def __init__(self):
        pass

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        pass

class TournamentSelection(SelectionOperator):
    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        chosen_sorted = sorted(chosen, key=lambda x: x.ff, reverse=True)
        return chosen_sorted[0].net.copy()

# TODO - A - test
class TournamentSelectionSized(SelectionOperator):
    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        size_sorted = sorted(chosen, key=lambda x: x.net.neuron_count)
        chosen_sorted = sorted(size_sorted, key=lambda x: x.ff, reverse=True)
        return chosen_sorted[0].net.copy()

class RoulletteSelection(SelectionOperator):
    def __init__(self):
        super().__init__()

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        sum_fit = sum([val_pop[i].ff for i in range(len(val_pop))])
        roulette = [[val_pop[i].net, val_pop[i].ff/sum_fit] for i in range(len(val_pop))]
        for i in range(1, len(roulette)):
            roulette[i][1] += roulette[i - 1][1]
        rr = random.random()
        for i in range(len(roulette)):
            if rr < roulette[i][1]:
                return roulette[i][0].copy()