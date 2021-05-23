from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, layerCount: (int, int), neuronCount: (int, int), actFuns: [ActFun],
                 lossFun: [LossFun]):
        self.layerCountMin = layerCount[0]
        self.layerCountMax = layerCount[1]
        self.neuronCountMin = neuronCount[0]
        self.neuronCountMax = neuronCount[1]

        self.actFunSet = actFuns
        self.lossFunSet = lossFun