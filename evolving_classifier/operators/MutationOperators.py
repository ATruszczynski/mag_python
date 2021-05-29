import random

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.HyperparameterRange import HyperparameterRange
from utility.Mut_Utility import *
from utility.Utility import *


class MutationOperator():
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: AnnPoint2, pm: float, radius: float) -> AnnPoint2:
        pass

class SomeStructMutationOperator(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: AnnPoint2, pm: float, radius: float) -> AnnPoint2:
        point = point.copy()

        # Zmień liczbę layerów
        if random.random() < pm:
            current = len(point.hidden_neuron_counts)
            minhl = max(current - 1, self.hrange.layerCountMin)
            maxhl = min(current + 1, self.hrange.layerCountMax)
            new = try_choose_different(current, list(range(minhl, maxhl + 1)))

            point = change_amount_of_layers(point=point, demanded=new, hrange=self.hrange)

        # Zmień county neuronów
        for i in range(len(point.hidden_neuron_counts)):
            if random.random() < pm:
                current = point.hidden_neuron_counts[i]
                new = try_choose_different(current, list(range(self.hrange.neuronCountMin, self.hrange.neuronCountMax + 1))) # TODO tu można wprowadzić radius
                point = change_neuron_count_in_layer(point=point, layer=i, demanded=new)

        # Zmień funkcje
        for i in range(len(point.weights)):
            if random.random() < pm:
                current = point.activation_functions[i]
                new = try_choose_different(current, self.hrange.actFunSet)
                point.activation_functions[i] = new.copy()

        return point

class SomeWBMutationOperator(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: AnnPoint2, pm: float, radius: float) -> AnnPoint2:
        point = point.copy()

        # Zmień wagi
        for i in range(len(point.weights)): #TODO can be made faster probably
            for r in range(point.weights[i].shape[0]):
                for c in range(point.weights[i].shape[1]):
                    if random.random() < pm:
                        point.weights[i][r, c] += random.gauss(0, radius)
        # Zmień biasy
        for i in range(len(point.biases)):#TODO can be made faster probably
            for r in range(point.biases[i].shape[0]):
                if random.random() < pm:
                    point.biases[i][r] += random.gauss(0, radius)

        return point