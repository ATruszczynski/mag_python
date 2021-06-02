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
            current = point.hiddenLayerCount
            minhl = max(current - 1, self.hrange.layerCountMin)
            maxhl = min(current + 1, self.hrange.layerCountMax)
            point.hiddenLayerCount = try_choose_different(point.hiddenLayerCount, range(minhl, maxhl))

        if random.random() < pm:
            point.neuronCount = get_in_radius(point.neuronCount, self.hrange.neuronCountMin, self.hrange.neuronCountMax, radius)

        if random.random() < pm * radius:
            point.actFun = try_choose_different(point.actFun, self.hrange.actFunSet)

        if random.random() < pm * radius:
            point.aggrFun = try_choose_different(point.aggrFun, self.hrange.aggrFunSet)

        if random.random() < pm * radius:
            point.lossFun = try_choose_different(point.lossFun, self.hrange.lossFunSet)

        if random.random() < pm:
            point.learningRate = get_in_radius(point.learningRate, self.hrange.learningRateMin, self.hrange.learningRateMax, radius)

        if random.random() < pm:
            point.momCoeff = get_in_radius(point.momCoeff, self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax, radius)

        if random.random() < pm:
            point.batchSize = get_in_radius(point.batchSize, self.hrange.batchSizeMin, self.hrange.batchSizeMax, radius)

        return
