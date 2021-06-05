import random
import threading
from math import sqrt, ceil
from statistics import mean

from ann_point import AnnPoint2
from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from utility.Utility import *
# import tensorflow as tf

class FeedForwardNeuralNetwork:
    def __init__(self, neuronCounts: [int], actFun: [ActFun], lossFun: LossFun, learningRate: float, momCoeff: float,
                 batchSize: float, seed: int):
        self.neuronCounts = []

        for i in range(len(neuronCounts)):
            self.neuronCounts.append(neuronCounts[i])

        self.actFuns = [None]
        for i in range(len(actFun)):
            self.actFuns.append(actFun[i].copy())
        self.lossFun = lossFun.copy()
        self.learningRate = 10 ** learningRate
        self.momCoeff = 10 ** momCoeff
        self.batchSize = 2 ** batchSize

        random.seed(seed)
        np.random.seed(seed)

        self.layerCount = len(self.neuronCounts)

        self.weights = None
        self.biases = None
        self.weight_mom = None
        self.biases_mom = None
        self.inp = None
        self.act = None

        self.init_wb()
        self.cm_hist = []

    def init_wb(self):
        self.weights = self.layerCount * [np.empty((0, 0))]
        self.biases = self.layerCount * [np.empty((0, 0))]
        self.inp = self.layerCount * [np.empty((0, 0))]
        self.act = self.layerCount * [np.empty((0, 0))]

        for i in range(1, self.layerCount):
            self.weights[i] = get_Xu_matrix((self.neuronCounts[i], self.neuronCounts[i - 1]))
            self.biases[i] = np.zeros((self.neuronCounts[i], 1))

        self.weight_mom = self.get_empty_weights()
        self.biases_mom = self.get_empty_biases()

    def run(self, arg: np.ndarray) -> np.ndarray:
        self.act[0] = arg

        for i in range(1, self.layerCount):
            act_prev = self.act[i - 1]
            self.inp[i] = np.dot(self.weights[i], act_prev) + self.biases[i]
            self.act[i] = self.actFuns[i].compute(self.inp[i])

        return self.act[self.layerCount - 1].copy()

    def train(self, inputs: [np.ndarray], outputs: [np.ndarray], epochs: int):
        batchSize = ceil(len(outputs) * self.batchSize) #TODO test czy dobrze dzielone na batche w testowaniu
        batches = divideIntoBatches(inputs, outputs, batchSize)

        for e in range(0, epochs):
            for b in range(0, len(batches)):
                weight_change, biases_change = self.get_grad(batches[b])

                weight_step = self.get_empty_weights()
                bias_step = self.get_empty_biases()

                for i in range(1, len(self.weight_mom)):
                    self.weight_mom[i] = self.momCoeff * self.weight_mom[i]
                    self.weight_mom[i] += self.learningRate * weight_change[i]
                    weight_step[i] = self.weight_mom[i].copy()

                    self.biases_mom[i] = self.momCoeff * self.biases_mom[i]
                    self.biases_mom[i] += self.learningRate * biases_change[i]
                    bias_step[i] = self.biases_mom[i].copy()

                for i in range(1, len(self.weights)):
                    self.weights[i] -= weight_step[i]
                    self.biases[i] -= bias_step[i]

    def test(self, test_input: [np.ndarray], test_output: [np.ndarray]) -> [float, float, float, np.ndarray]:
        out_size = self.neuronCounts[len(self.neuronCounts) - 1]
        confusion_matrix = np.zeros((out_size, out_size))

        for i in range(len(test_output)):
            net_result = self.run(test_input[i])
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(test_output[i])
            confusion_matrix[corr_class, pred_class] += 1

        return [accuracy(confusion_matrix), average_precision(confusion_matrix), average_recall(confusion_matrix), confusion_matrix]

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

    def get_grad(self, batch):
        weight_change = self.get_empty_weights()
        biases_change = self.get_empty_biases()

        cm = np.zeros((self.neuronCounts[-1], self.neuronCounts[-1]))

        for data in batch:
            net_input = data[0]
            net_output = data[1]

            weight_grad = self.get_empty_weights()
            bias_grad = self.get_empty_biases()
            act_grad = self.get_empty_activations()

            for l in range(self.layerCount - 1, 0, -1):
                if l is self.layerCount - 1:
                    net_result = self.run(net_input)
                    act_grad[l] = self.lossFun.computeDer(net_result, net_output)

                    pred_class = np.argmax(net_result)
                    corr_class = np.argmax(net_output)
                    cm[corr_class, pred_class] += 1
                else:
                    act_grad[l] = np.dot(bias_grad[l + 1].T, self.weights[l + 1]).T


                bias_grad[l] = np.dot(self.actFuns[l].computeDer(self.inp[l]), act_grad[l])
                weight_grad[l] = np.dot(bias_grad[l], self.act[l - 1].T)

            for i in range(1, len(weight_grad)):
                weight_change[i] += weight_grad[i]
                biases_change[i] += bias_grad[i]

        batch_count = len(batch)

        for ind in range(1, len(weight_change)):
            weight_change[ind] /= batch_count
            biases_change[ind] /= batch_count

        self.cm_hist.append(cm)

        return weight_change, biases_change


def network_from_point(point: AnnPoint, seed: int):
    return FeedForwardNeuralNetwork \
        (neuronCounts=point.neuronCounts, actFun=point.actFuns,
         lossFun=point.lossFun.copy(), learningRate=point.learningRate, momCoeff=point.momCoeff,
         batchSize=point.batchSize, seed=seed)

def accuracy(confusion_matrix: np.ndarray):
    tot_sum = np.sum(confusion_matrix)
    diag_sum = np.sum(np.diag(confusion_matrix))

    accuracy = diag_sum / tot_sum

    return accuracy

def average_precision(conf_matrix):
    return np.average(get_precisions(conf_matrix))

def get_precisions(conf_matrix) -> [float]:
    col_sum = np.sum(conf_matrix, axis=0)
    row_sum = np.sum(conf_matrix, axis=1)
    diag = np.diag(conf_matrix)

    class_prec = []

    for i in range(len(col_sum)):
        if row_sum[i] > 0:
            if col_sum[i] > 0:
                class_prec.append(diag[i] / col_sum[i])
            else:
                class_prec.append(0)

    return class_prec


def average_recall(conf_matrix):
    return np.average(get_recalls(conf_matrix))

def get_recalls(conf_matrix) -> [float]:
    row_sums = np.sum(conf_matrix, axis=1)
    diag = np.diag(conf_matrix)

    class_recalls = []

    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            class_recalls.append(diag[i] / row_sums[i])

    return class_recalls


def efficiency(conf_matrix):
    acc = accuracy(conf_matrix)
    prec = average_precision(conf_matrix)
    rec = average_recall(conf_matrix)

    return mean([acc, prec, rec])


def average_f1_score(conf_matrix):
    precisions = get_precisions(conf_matrix)
    recall = get_recalls(conf_matrix)

    f1 = []
    for i in range(len(precisions)):
        prec_inv = 1 / precisions[i]
        rec_inv = 1 / recall[i]
        f1.append(2 / (rec_inv + prec_inv))

    return mean(f1)



