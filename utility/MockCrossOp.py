from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.LsmCrossoverOperator import CrossoverOperator
from evolving_classifier.operators.LsmMutationOperator import MutationOperator
from neural_network.LsmNetwork import LsmNetwork


class MockCO(CrossoverOperator):
    def __init__(self):
        pass

    def crossover(self, pointA: LsmNetwork, pointB: LsmNetwork) -> [LsmNetwork, LsmNetwork]:
        pass

