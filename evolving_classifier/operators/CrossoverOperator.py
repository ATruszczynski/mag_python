import random

from ann_point.AnnPoint2 import AnnPoint2, point_from_layer_tuples
from utility.Mut_Utility import fix_layer_sizes
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

        # for i in range(1, len(layersA)): # TODO to chyba powinna być osobna funkcja
        #     layer = layersA[i]
        #     pre_layer = layersA[i - 1]
        #     if pre_layer[1] != layer[3].shape[1]:
        #         layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
        #         layer[4] = np.zeros((layer[1], 1))
        #     layersA[i] = layer
        #
        # for i in range(1, len(layersB)): # TODO to chyba powinna być osobna funkcja
        #     layer = layersB[i]
        #     pre_layer = layersB[i - 1]
        #     if pre_layer[1] != layer[3].shape[1]:
        #         layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
        #         layer[4] = np.zeros((layer[1], 1))
        #     layersB[i] = layer

        layersA = fix_layer_sizes(layersA)
        layersB = fix_layer_sizes(layersB)

        pointA = point_from_layer_tuples(layersA)
        pointB = point_from_layer_tuples(layersB)

        return [pointA, pointB]