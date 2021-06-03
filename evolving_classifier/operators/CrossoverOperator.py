import random

from ann_point.AnnPoint import AnnPoint
from ann_point.AnnPoint2 import AnnPoint2, point_from_layer_tuples
from utility.Mut_Utility import *
from utility.Utility import get_Xu_matrix
import numpy as np


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint]:
        pass

class SimpleCrossoverOperator:
    def __init__(self, swap_prob: float = 0.5):
        self.swap_prob = swap_prob
        pass

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint) -> [AnnPoint, AnnPoint]:
        if len(pointA.neuronCounts) <= len(pointB.neuronCounts):
            pointA = pointA.copy()
            pointB = pointB.copy()
        else:
            tmp = pointA.copy()
            pointA = pointB.copy()
            pointB = tmp

        swap_options = []
        for i in range(1, len(pointA.neuronCounts)):
            for j in range(i, len(pointB.neuronCounts)):
                swap_options.append((i, j))

        swapInds = swap_options[random.randint(0, len(swap_options) - 1)]
        swapAInd = swapInds[0]
        swapBInd = swapInds[1]

        layersA = pointA.get_layer_struct()
        layersB = pointB.get_layer_struct()

        tmp = layersA[swapAInd:]
        layersA[swapAInd:] = layersB[swapBInd:]
        layersB[swapBInd:] = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.lossFun.copy()
            pointA.lossFun = pointB.lossFun.copy()
            pointB.lossFun = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.learningRate
            pointA.learningRate = pointB.learningRate
            pointB.learningRate = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.momCoeff
            pointA.momCoeff = pointB.momCoeff
            pointB.momCoeff = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.batchSize
            pointA.batchSize = pointB.batchSize
            pointB.batchSize = tmp

        pointA = point_from_layers(layersA, lossFun=pointA.lossFun, learningRate=pointA.learningRate, momCoeff=pointA.momCoeff, batchSize=pointA.batchSize)
        pointB = point_from_layers(layersB, lossFun=pointB.lossFun, learningRate=pointB.learningRate, momCoeff=pointB.momCoeff, batchSize=pointB.batchSize)

        return pointA, pointB

class LayerSwapCrossoverOperator:
    def __init__(self, swap_prob: float = 0.5):
        self.swap_prob = swap_prob
        pass

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint) -> [AnnPoint, AnnPoint]:
        layersA = pointA.get_layer_struct()
        layersB = pointB.get_layer_struct()

        if random.random() < self.swap_prob:
            tmp = layersA
            layersA = layersB
            layersB = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.lossFun.copy()
            pointA.lossFun = pointB.lossFun.copy()
            pointB.lossFun = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.learningRate
            pointA.learningRate = pointB.learningRate
            pointB.learningRate = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.momCoeff
            pointA.momCoeff = pointB.momCoeff
            pointB.momCoeff = tmp

        if random.random() < self.swap_prob:
            tmp = pointA.batchSize
            pointA.batchSize = pointB.batchSize
            pointB.batchSize = tmp

        pointA = point_from_layers(layersA, lossFun=pointA.lossFun, learningRate=pointA.learningRate, momCoeff=pointA.momCoeff, batchSize=pointA.batchSize)
        pointB = point_from_layers(layersB, lossFun=pointB.lossFun, learningRate=pointB.learningRate, momCoeff=pointB.momCoeff, batchSize=pointB.batchSize)

        return pointA, pointB




