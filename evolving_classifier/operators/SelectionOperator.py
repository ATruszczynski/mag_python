from ann_point.AnnPoint2 import AnnPoint2
from utility.Utility import choose_without_repetition, AnnPoint


class SelectionOperator:
    def __init__(self, count: int):
        self.count = count

    def select(self, val_pop: [[AnnPoint2, float]]) -> AnnPoint2:
        pass


class TournamentSelection(SelectionOperator):
    def __init__(self, count: int):
        super().__init__(count)

    def select(self, val_pop: [[AnnPoint, float]]) -> AnnPoint:
        chosen = choose_without_repetition(options=val_pop, count=self.count)
        chosen_sorted = sorted(chosen, key=lambda x: x[1], reverse=True)
        return chosen_sorted[0][0]