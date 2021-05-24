from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, neuronCount: int, actFuns: [ActFun]):
        self.neuronCount = neuronCount
        self.actFunSet = actFuns
        self.maxAbsWei = 1
        self.maxAbsBia = 1