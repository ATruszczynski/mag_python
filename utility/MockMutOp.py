from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.LsmMutationOperator import MutationOperator
from neural_network.LsmNetwork import LsmNetwork


class MockMO(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: LsmNetwork) -> LsmNetwork:
        pass

