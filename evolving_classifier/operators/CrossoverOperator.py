import random

from ann_point.AnnPoint2 import AnnPoint2, point_from_layer_tuples
from utility.Mut_Utility import *
from utility.Utility import get_Xu_matrix
import numpy as np


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: AnnPoint2, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint2]:
        pass

class SomeCrossoverOperator(CrossoverOperator):
    def __init__(self):
        super().__init__()

    def crossover(self, pointA: AnnPoint2, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint2]:
        pointA = pointA.copy()
        pointB = pointB.copy()

        if len(pointA.hidden_neuron_counts) < len(pointB.hidden_neuron_counts):
            tmp = pointA
            pointA = pointB
            pointB = tmp

        layersA = pointA.into_numbered_layer_tuples()
        layersB = pointB.into_numbered_layer_tuples()

        choices = []
        for i in range(1, len(layersA) - 1):
            for j in range(1, len(layersB) - 1):
                choices.append((i, j))
        choices.append((len(layersA) - 1, len(layersB) - 1))

        choice = choices[random.randint(0, len(choices) - 1)]

        layAInd = choice[0]
        layBInd = choice[1]

        tmp = layersA[layAInd][:3]
        layersA[layAInd] = layersB[layBInd][:3]
        layersB[layBInd] = tmp

        layersA = fix_layer_sizes(layersA)
        layersB = fix_layer_sizes(layersB)

        pointA = point_from_layer_tuples(layersA)
        pointB = point_from_layer_tuples(layersB)

        return [pointA, pointB]


class MinimalDamageCrossoverOperator(CrossoverOperator):
    def __init__(self):
        super().__init__()

    def crossover(self, pointA: AnnPoint2, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint2]:
        pointA = pointA.copy()
        pointB = pointB.copy()

        if len(pointA.hidden_neuron_counts) < len(pointB.hidden_neuron_counts):
            tmp = pointA
            pointA = pointB
            pointB = tmp

        layersA = pointA.into_numbered_layer_tuples()
        layersB = pointB.into_numbered_layer_tuples()

        choices = []
        for i in range(1, len(layersA) - 1):
            for j in range(1, len(layersB) - 1):
                choices.append((i, j))
        choices.append((len(layersA) - 1, len(layersB) - 1))

        choice = choices[random.randint(0, len(choices) - 1)]

        layAInd = choice[0]
        layBInd = choice[1]

        tmp = layersA[layAInd]
        layersA[layAInd] = layersB[layBInd]
        layersB[layBInd] = tmp

        layersA = resize_given_layer(layAInd, layersA)
        layersB = resize_given_layer(layBInd, layersB)

        pointA = point_from_layer_tuples(layersA)
        pointB = point_from_layer_tuples(layersB)

        return [pointA, pointB]