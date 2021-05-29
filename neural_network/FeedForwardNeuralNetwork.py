import random
from math import sqrt, ceil

from ann_point import AnnPoint2
from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from utility.Utility import *
# import tensorflow as tf

class FeedForwardNeuralNetwork:
    def __init__(self, inputSize: int, outputSize: int, hidden_neuron_counts: [float], actFuns: [ActFun], weights: [np.ndarray], biases: [np.ndarray], seed: int):
        self.neuronCounts = [inputSize]

        for i in range(len(hidden_neuron_counts)):
            self.neuronCounts.append(hidden_neuron_counts[i])

        self.neuronCounts.append(outputSize)

        random.seed(seed)# TODO check if this works correctly in multithread

        self.layerCount = len(hidden_neuron_counts) + 2
        self.actFuns = [None]
        for i in range(len(actFuns)):
            self.actFuns.append(actFuns[i].copy())

        self.weights = [np.empty((0, 0))]
        self.biases = [np.empty((0, 0))]
        for i in range(len(weights)):
            self.weights.append(weights[i].copy()) # TODO check if deep copy
            self.biases.append(biases[i].copy()) # TODO check if deep copy

        self.inp = self.layerCount * [np.empty((0, 0))]
        self.act = self.layerCount * [np.empty((0, 0))]

        self.weight_mom = self.get_empty_weights()
        self.biases_mom = self.get_empty_biases()
        self.lossFun = CrossEntropy()
        self.momCoeffL = 0.01
        self.learningRate = 0.01

    # def init_wb(self):
    #     self.weights = self.layerCount * [np.empty((0, 0))]
    #     self.biases = self.layerCount * [np.empty((0, 0))]
    #     self.inp = self.layerCount * [np.empty((0, 0))]
    #     self.act = self.layerCount * [np.empty((0, 0))]
    #
    #     for i in range(1, self.layerCount):
    #         self.weights[i] = np.zeros((self.neuronCounts[i], self.neuronCounts[i - 1]))
    #         self.biases[i] = np.zeros((self.neuronCounts[i], 1))
    #
    #         for r in range(0, self.weights[i].shape[0]):
    #             for c in range(0, self.weights[i].shape[1]):
    #                 self.weights[i][r, c] = random.gauss(0, 1 / sqrt(self.neuronCounts[i - 1]))
    #
    #     self.weight_mom = self.get_empty_weights()
    #     self.biases_mom = self.get_empty_biases()

    def run(self, arg: np.ndarray) -> np.ndarray:
        self.act[0] = arg

        for i in range(1, self.layerCount):
            act_prev = self.act[i - 1]
            self.inp[i] = np.dot(self.weights[i], act_prev) + self.biases[i]
            self.act[i] = self.actFuns[i].compute(self.inp[i])

        return self.act[self.layerCount - 1].copy()

    def train(self, inputs: [np.ndarray], outputs: [np.ndarray], epochs: int):
        batchSize = 0.05
        batchSize = ceil(len(outputs) * batchSize)
        batches = divideIntoBatches(inputs, outputs, batchSize)

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
                            act_grad[l] = np.dot(bias_grad[l + 1].T, self.weights[l + 1]).T


                        bias_grad[l] = np.dot(self.actFuns[l].computeDer(self.inp[l]), act_grad[l])
                        weight_grad[l] = np.dot(bias_grad[l], self.act[l - 1].T)

                    for i in range(1, len(weight_grad)):
                        weight_change[i] += weight_grad[i]
                        biases_change[i] += bias_grad[i]

                batch_count = len(batches[b])

                for ind in range(1, len(weight_change)):
                    weight_change[ind] /= batch_count
                    biases_change[ind] /= batch_count


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

    def test(self, test_input: [np.ndarray], test_output: [np.ndarray], lossFun: LossFun = None) -> [float, float, float, np.ndarray]:
        out_size = self.neuronCounts[len(self.neuronCounts) - 1]
        confusion_matrix = np.zeros((out_size, out_size))

        result = 0

        for i in range(len(test_output)):
            net_result = self.run(test_input[i])
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(test_output[i])
            confusion_matrix[corr_class, pred_class] += 1
            if lossFun is not None:
                result += lossFun.compute(net_result, test_output[i])

        # TODO test acc, recall and precision

        return [accuracy(confusion_matrix), average_precision(confusion_matrix), average_recall(confusion_matrix), confusion_matrix, result]

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


def network_from_point(point: AnnPoint2, seed: int):
    return FeedForwardNeuralNetwork \
        (inputSize=point.input_size, outputSize=point.output_size, hidden_neuron_counts=point.hidden_neuron_counts, actFuns=point.activation_functions, weights=point.weights, biases=point.biases, seed=seed)

def accuracy(confusion_matrix: np.ndarray):
    tot_sum = np.sum(confusion_matrix)
    diag_sum = np.sum(np.diag(confusion_matrix))

    accuracy = diag_sum / tot_sum

    return accuracy

def average_precision(conf_matrix):
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

    return np.average(class_prec)

def average_recall(conf_matrix):
    row_sums = np.sum(conf_matrix, axis=1)
    diag = np.diag(conf_matrix)

    class_recalls = []

    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            class_recalls.append(diag[i] / row_sums[i])

    return np.average(class_recalls)


