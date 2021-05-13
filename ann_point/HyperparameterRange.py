from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, layerCount: (int, int), neuronCount: (float, float), actFuns: [ActFun], aggrFun: [ActFun],
                 lossFun: [LossFun], learningRate: (float, float), momentumCoeff: (float, float)):
        self.layerCountMin = layerCount[0]
        self.layerCountMax = layerCount[1]
        self.neuronCountMin = neuronCount[0]
        self.neuronCountMax = neuronCount[1]
        self.learningRateMin = learningRate[0]
        self.learningRateMax = learningRate[1]
        self.momentumCoeffMin = momentumCoeff[0]
        self.momentumCoeffMax = momentumCoeff[1]

        self.actFunSet = actFuns
        self.aggrFunSet = aggrFun
        self.lossFunSet = lossFun