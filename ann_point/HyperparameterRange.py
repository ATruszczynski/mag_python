from ann_point.Functions import *

class HyperparameterRange:
    # def __init__(self, layerCount: (int, int), neuronCount: (int, int), actFuns: [ActFun],
    #              lossFun: [LossFun], learningRate: (float, float), momentumCoeff: (float, float), batchSize: (float, float)):
    def __init__(self, init_wei: (float, float), init_bia: (float, float), it: (int, int), actFuns: [ActFun]):
        # self.hiddenLayerCountMin = layerCount[0]
        # self.hiddenLayerCountMax = layerCount[1]
        # self.neuronCountMin = neuronCount[0]
        # self.neuronCountMax = neuronCount[1]
        # self.learningRateMin = learningRate[0]
        # self.learningRateMax = learningRate[1]
        # self.momentumCoeffMin = momentumCoeff[0]
        # self.momentumCoeffMax = momentumCoeff[1]
        # self.batchSizeMin = batchSize[0]
        # self.batchSizeMax = batchSize[1]

        self.min_init_wei = init_wei[0]
        self.max_init_wei = init_wei[1]
        self.min_init_bia = init_bia[0]
        self.max_init_bia = init_bia[1]
        self.min_it = it[0]
        self.max_it = it[1]

        self.actFunSet = actFuns
