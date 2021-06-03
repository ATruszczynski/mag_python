from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, layerCount: (int, int), neuronCount: (int, int), actFuns: [ActFun],
                 lossFun: [LossFun], learningRate: (float, float), momentumCoeff: (float, float), batchSize: (float, float)):
        self.hiddenLayerCountMin = layerCount[0]
        self.hiddenLayerCountMax = layerCount[1]
        self.neuronCountMin = neuronCount[0]
        self.neuronCountMax = neuronCount[1]
        self.learningRateMin = learningRate[0]
        self.learningRateMax = learningRate[1]
        self.momentumCoeffMin = momentumCoeff[0]
        self.momentumCoeffMax = momentumCoeff[1]
        self.batchSizeMin = batchSize[0]
        self.batchSizeMax = batchSize[1]

        self.actFunSet = actFuns
        self.lossFunSet = lossFun