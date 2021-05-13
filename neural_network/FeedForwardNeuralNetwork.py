import random
from math import sqrt, ceil

from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *

class FeedForwardNeuralNetwork:
    def __init__(self, inputSize: int, outputSize: int, hiddenLayerCount: int, neuronCount: int,
                 actFun: ActFun, aggrFun: ActFun, lossFun: LossFun, learningRate: float, momCoeffL: float, seed: int):
        self.neuronCounts = [inputSize]

        for i in range(0, hiddenLayerCount):
            self.neuronCounts.append(neuronCount)

        self.neuronCounts.append(outputSize)

        self.actFun = actFun
        self.aggrFun = aggrFun
        self.lossFun = lossFun
        self.learningRate = learningRate
        self.momCoeffL = momCoeffL

        random.seed(seed) # TODO check if this works correctly in multithread

        self.layerCount = hiddenLayerCount + 2
        self.actFuns = []
        for i in range(1, self.layerCount):
            if i is not self.layerCount - 1:
                self.actFuns.append(actFun.copy())
            else:
                self.actFuns.append(aggrFun.copy())

        self.weights = None
        self.biases = None
        self.weight_mom = None
        self.biases_mom = None
        self.inp = None
        self.act = None

        self.init_wb()

    def init_wb(self):
        self.weights = self.layerCount * [np.empty((0, 0))]
        self.biases = self.layerCount * [np.empty((0, 0))]
        self.inp = self.layerCount * [np.empty((0, 0))]
        self.act = self.layerCount * [np.empty((0, 0))]

        for i in range(1, self.layerCount):
            self.weights[i] = np.zeros((self.neuronCounts[i], self.neuronCounts[i - 1]))
            self.biases[i] = np.zeros((self.neuronCounts[i], 1))

            for r in range(0, self.weights[i].shape[0]):
                for c in range(0, self.weights[i].shape[1]):
                    self.weights[i][r, c] = random.gauss(0, 1 / sqrt(self.neuronCounts[i - 1]))

        self.weight_mom = self.get_empty_weights()
        self.biases_mom = self.get_empty_biases()

    def run(self, arg: np.ndarray) -> np.ndarray:
        self.act[0] = arg

        for i in range(1, self.layerCount):
            act_prev = self.act[i - 1]

            self.inp[i] = np.dot(self.weights[i], act_prev) + self.biases[i]
            self.act[i] = self.actFuns[i].compute(self.inp[i])

        return self.act[self.layerCount - 1].copy()

    def train(self, inputs: [np.ndarray], outputs: [np.ndarray], epochs: int, batchSize: int):
        batches = self.divideIntoBatches(inputs, outputs, batchSize)

        for e in range(0, epochs):
            for b in range(0, len(batches)):
                weight_change = self.get_empty_weights()
                biases_change = self.get_empty_biases()

                for data in batches[b]:
                    net_input = data[0]
                    net_output = data[1]

                    weight_grad = self.get_empty_weights()
                    bias_grad = self.get_empty_biases()
                    act_grad = self.get_empty_activations()

                    for l in range(self.layerCount - 1, 0, -1):
                        if l is self.layerCount - 1:
                            net_result = self.run(net_input)
                            act_grad[l] = self.lossFun.computeDer(net_result, net_output)
                        else:
                            act_grad[l] = np.dot(bias_grad[l + 1].T, self.weights[l + 1])

                        bias_grad[l] = np.multiply(self.actFuns[l].computeDer(self.inp[l]), act_grad[l])
                        weight_grad[l] = np.dot(bias_grad[l], self.act[l - 1].T)

                    for i in range(1, len(weight_grad)):
                        weight_change[i] += weight_grad[i]
                        biases_change[i] += biases_change[i]

                weight_step = self.get_empty_weights()
                bias_step = self.get_empty_biases()

                for i in range(1, len(self.weight_mom)):
                    self.weight_mom[i] = self.momCoeffL * self.weight_mom[i]
                    self.weight_mom[i] += self.learningRate * weight_change[i]
                    weight_step[i] = self.weight_mom[i].copy()

                    self.biases_mom[i] = self.momCoeffL * self.biases_mom[i]
                    self.biases_mom[i] += self.learningRate * biases_change[i]
                    bias_step[i] = self.biases_mom[i].copy()

                for i in range(1, len(self.weights)):
                    self.weights[i] -= weight_step[i]
                    self.biases[i] -= bias_step[i]



    def divideIntoBatches(self, inputs: [np.ndarray], outputs: [np.ndarray], batchSize: int):
        count = len(inputs)
        batchCount = ceil(count/batchSize)

        batches = []

        for i in range(0, batchCount):
            batch = []
            for j in range(i * batchSize, min((i + 1) * batchSize, count)):
                batch.append((inputs[i], outputs[i]))
            batches.append(batch)

        return batches

    def get_empty_weights(self):
        result = [np.empty((0, 0))]
        for i in range(1, self.layerCount):
            result.append(np.zeros(self.weights[i].shape))

        return result

    def get_empty_biases(self):
        result = [np.empty((0, 0))]
        for i in range(1, self.layerCount):
            result.append(np.zeros(self.biases[i].shape))

        return result

    def get_empty_activations(self):
        result = [np.empty((0, 0))]
        for i in range(1, self.layerCount):
            result.append(np.zeros(self.act[i].shape))

        return result


def network_from_point(point: AnnPoint):
    return FeedForwardNeuralNetwork \
        (inputSize=point.inputSize, outputSize=point.outputSize, hiddenLayerCount=point.hiddenLayerCount,
         neuronCount=point.neuronCount, actFun=point.actFun.copy(), aggrFun=point.aggrFun.copy(),
         lossFun=point.lossFun.copy(), learningRate=point.learningRate, momCoeffL=point.momCoeff)





