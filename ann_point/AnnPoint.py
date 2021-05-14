from ann_point.Functions import *

class AnnPoint():
    def __init__(self, inputSize: int, outputSize: int, hiddenLayerCount: int, neuronCount: float,
                 actFun: ActFun, aggrFun: ActFun, lossFun: LossFun, learningRate: float, momCoeff: float):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayerCount = hiddenLayerCount
        self.neuronCount = neuronCount
        self.actFun = actFun.copy()
        self.aggrFun = aggrFun.copy()
        self.lossFun = lossFun.copy()
        self.learningRate = learningRate
        self.momCoeff = momCoeff

    def copy(self):
        result = AnnPoint(inputSize=self.inputSize, outputSize=self.outputSize, hiddenLayerCount=self.hiddenLayerCount,
                          neuronCount=self.neuronCount, actFun=self.actFun.copy(), aggrFun=self.aggrFun.copy(),
                          lossFun=self.lossFun.copy(), learningRate=self.learningRate, momCoeff=self.momCoeff)

        return result

    def to_string(self):
        result = ""
        result += str(self.inputSize) + "|" + str(self.outputSize) + "|" + str(self.hiddenLayerCount) + "|" + \
                  str(round(self.neuronCount, 2)) + "|" +str(self.actFun.to_string()) + "|" + str(self.aggrFun.to_string()) + "|" + \
                  str(self.lossFun.to_string()) + "|" + str(round(self.learningRate, 2)) + "|" + str(round(self.momCoeff, 2)) + "|"

        return result