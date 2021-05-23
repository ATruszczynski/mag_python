from math import ceil

from ann_point.Functions import *

class AnnPoint():
    def __init__(self, inputSize: int, outputSize: int, hiddenLayerCount: int, neuronCount: float,
                 actFun: ActFun, aggrFun: ActFun, lossFun: LossFun, learningRate: float, momCoeff: float, batchSize: float):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayerCount = hiddenLayerCount
        self.neuronCount = neuronCount
        self.actFun = actFun.copy()
        self.aggrFun = aggrFun.copy()
        self.lossFun = lossFun.copy()
        self.learningRate = learningRate
        self.momCoeff = momCoeff
        self.batchSize = batchSize

    def copy(self):
        result = AnnPoint(inputSize=self.inputSize, outputSize=self.outputSize, hiddenLayerCount=self.hiddenLayerCount,
                          neuronCount=self.neuronCount, actFun=self.actFun.copy(), aggrFun=self.aggrFun.copy(),
                          lossFun=self.lossFun.copy(), learningRate=self.learningRate, momCoeff=self.momCoeff, batchSize=self.batchSize)

        return result

    def to_string(self):
        result = ""
        result += "|" + str(self.inputSize) + "|" + str(self.outputSize) + "|" + str(self.hiddenLayerCount) + "|" + \
                  str(round(self.neuronCount, 2)) + "|" +str(self.actFun.to_string()) + "|" + str(self.aggrFun.to_string()) + "|" + \
                  str(self.lossFun.to_string()) + "|" + str(round(self.learningRate, 2)) + "|" + str(round(self.momCoeff, 2)) + "|" \
                  + str(round(self.batchSize, 2)) + "|"

        return result

    def to_string_full(self): # TODO test
        result = ""
        result += "|" + str(self.inputSize) + "|" + str(self.outputSize) + "|" + str(self.hiddenLayerCount) + "|" + \
                  str(self.neuronCount) + "|" +str(self.actFun.to_string()) + "|" + str(self.aggrFun.to_string()) + "|" + \
                  str(self.lossFun.to_string()) + "|" + str(self.learningRate) + "|" + str(self.momCoeff) + "|" \
                  + str(self.batchSize) + "|"

        return result

    def size(self):
        result = 0
        neuron_counts = [self.inputSize]
        for i in range(self.hiddenLayerCount):
            neuron_counts.append(ceil(2 ** self.neuronCount))
        neuron_counts.append(self.outputSize)

        for i in range(len(neuron_counts) - 1):
            result += neuron_counts[i] * neuron_counts[i + 1]

        return result

