from math import ceil

from ann_point.Functions import *

class AnnPoint():
    def __init__(self, neuronCounts: [int],
                 actFuns: [ActFun], lossFun: LossFun, learningRate: float, momCoeff: float, batchSize: float):
        self.neuronCounts = []
        for i in range(len(neuronCounts)):
            self.neuronCounts.append(neuronCounts[i])
        self.actFuns = []
        for i in range(len(actFuns)):
            self.actFuns.append(actFuns[i].copy())
        self.lossFun = lossFun.copy()
        self.learningRate = learningRate
        self.momCoeff = momCoeff
        self.batchSize = batchSize

    def copy(self):
        neuronCounts = []
        for i in range(len(self.neuronCounts)):
            neuronCounts.append(self.neuronCounts[i])
        actFuns = []
        for i in range(len(self.actFuns)):
            actFuns.append(self.actFuns[i].copy())
        result = AnnPoint(neuronCounts=neuronCounts, actFuns=actFuns,
                          lossFun=self.lossFun.copy(), learningRate=self.learningRate, momCoeff=self.momCoeff, batchSize=self.batchSize)

        return result

    def to_string(self):
        result = ""

        layer_string = "|" + str(self.neuronCounts[0]) + "|"
        for i in range(1, len(self.neuronCounts)):
            layer_string += f"{self.actFuns[i - 1].to_string()}|{str(self.neuronCounts[i])}|"


        result += layer_string + \
                  str(self.lossFun.to_string()) + "|" + str(round(self.learningRate, 2)) + "|" + str(round(self.momCoeff, 2)) + "|" \
                  + str(round(self.batchSize, 2)) + "|"

        return result

    def to_string_full(self): # TODO test
        result = ""
        layer_string = "|" + str(self.neuronCounts[0]) + "|"
        for i in range(1, len(self.neuronCounts)):
            layer_string += f"{self.actFuns[i - 1].to_string()}|{str(self.neuronCounts[i])}|"

        result += layer_string + \
                  str(self.lossFun.to_string()) + "|" + str(self.learningRate) + "|" + str(self.momCoeff) + "|" \
                  + str(self.batchSize) + "|"

        return result

    def size(self):
        result = 0

        for i in range(len(self.neuronCounts) - 1):
            result += self.neuronCounts[i] * self.neuronCounts[i + 1]

        return result

    def get_layer_struct(self) -> [[int, int, ActFun]]:
        result = [[0, self.neuronCounts[0], None]]
        for i in range(1, len(self.neuronCounts)):
            result.append([i, self.neuronCounts[i], self.actFuns[i - 1]])

        return result

def point_from_layers(layers: [[int, int, ActFun]], lossFun: LossFun, learningRate: float, momCoeff: float, batchSize: float) -> AnnPoint:
    neuronCounts = [l[1] for l in layers]
    actFuns = [l[2] for l in layers[1:]]

    return AnnPoint(neuronCounts=neuronCounts, actFuns=actFuns, lossFun=lossFun, learningRate=learningRate, momCoeff=momCoeff, batchSize=batchSize)

