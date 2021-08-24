from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.MutationOperators import MutationOperator
from neural_network.ChaosNet import ChaosNet


class MockMO(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: ChaosNet) -> ChaosNet:
        pass

