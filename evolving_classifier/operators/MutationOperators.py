import random

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.HyperparameterRange import HyperparameterRange
from utility.Mut_Utility import *
from utility.Utility import *


class MutationOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: AnnPoint, pm: float, radius: float) -> AnnPoint:
        pass

class SimpleMutationOperator():
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: AnnPoint, pm: float, radius: float) -> AnnPoint:
        point = point.copy()

        if random.random() < pm * radius:
            current = len(point.neuronCounts) - 2
            minhl = max(current - 1, self.hrange.hiddenLayerCountMin)
            maxhl = min(current + 1, self.hrange.hiddenLayerCountMax)

            new_lay_count = try_choose_different(current, range(minhl, maxhl + 1))
            diff = new_lay_count - current
            if diff > 0:
                point = add_layers(point=point, howMany=diff, hrange=self.hrange)
            elif diff < 0:
                point = remove_layers(point=point, howMany=-diff)

        for i in range(1, len(point.neuronCounts) - 1):
            if random.random() < pm:
                point.neuronCounts[i] = round(get_in_radius(point.neuronCounts[i], self.hrange.neuronCountMin, self.hrange.neuronCountMax, radius))

        for i in range(len(point.actFuns)):
            if random.random() < pm:
                point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)

        if random.random() < pm * radius:
            point.lossFun = try_choose_different(point.lossFun, self.hrange.lossFunSet)

        if random.random() < pm:
            point.learningRate = get_in_radius(point.learningRate, self.hrange.learningRateMin, self.hrange.learningRateMax, radius)

        if random.random() < pm:
            point.momCoeff = get_in_radius(point.momCoeff, self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax, radius)

        if random.random() < pm:
            point.batchSize = get_in_radius(point.batchSize, self.hrange.batchSizeMin, self.hrange.batchSizeMax, radius)

        return point
