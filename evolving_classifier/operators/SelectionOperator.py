import math
import random

from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint
from utility.Utility import choose_without_repetition


class SelectionOperator:
    def __init__(self):
        pass

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        pass

class TournamentSelection05(SelectionOperator):
    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        chosen_sorted = chosen
        for i in reversed(range(len(val_pop[0].ff))):
            chosen_sorted = sorted(chosen_sorted, key=lambda x: x.ff[i], reverse=True)

        for i in range(len(chosen_sorted) - 1):
            p = random.random()
            if p <= 0.95:
                return chosen_sorted[i].net.copy()

        return chosen_sorted[-1].net.copy()

class TournamentSelection(SelectionOperator):
    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        chosen_sorted = sorted(chosen, key=lambda x: x.ff, reverse=True)
        return chosen_sorted[0].net.copy()

# class TournamentSelectionSized(SelectionOperator):
#     def __init__(self, count: int):
#         super().__init__()
#         self.count = count
#
#     def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
#         chosen = choose_without_repetition(options=val_pop, count=self.count)
#         size_sorted = sorted(chosen, key=lambda x: x.net.neuron_count)
#         chosen_sorted = sorted(size_sorted, key=lambda x: x.ff, reverse=True)
#         return chosen_sorted[0].net.copy()

# class TournamentSelectionSized2(SelectionOperator):
#     def __init__(self, count: int):
#         super().__init__()
#         self.count = count
#
#     def select(self, val_pop: [CNDataPoint]) -> ChaosNet:
#         chosen = choose_without_repetition(options=val_pop, count=self.count)
#         chosen_sorted = sorted(chosen, key=lambda x: x.ff, reverse=True)
#
#         eps = 1e-10
#         perc = 0.5
#
#         bestff = abs(chosen_sorted[0].ff)
#         if math.isnan(bestff) or math.isinf(bestff):
#             chosen_filtered = chosen_sorted
#         else:
#             tol = perc * bestff
#             chosen_filtered = [cs for cs in chosen_sorted if abs(abs(cs.ff) - bestff) <= tol + eps]
#
#         chosen_size_sorted = sorted(chosen_filtered, key=lambda x: x.net.neuron_count)
#         return chosen_size_sorted[0].net.copy()

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