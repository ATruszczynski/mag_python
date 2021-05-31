from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, layerCount: (int, int), neuronCount: (int, int), actFuns: [ActFun], weiAbs: float = 1, biaAbs: float = 1):
        self.layerCountMin = layerCount[0]
        self.layerCountMax = layerCount[1]
        self.neuronCountMin = neuronCount[0]
        self.neuronCountMax = neuronCount[1]
        self.actFunSet = actFuns
        self.weiAbs = weiAbs
        self.biaAbs = biaAbs