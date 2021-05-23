import time

import numpy as np
from ann_point.Functions import ActFun
from utility.Utility import *


class LooseNetwork:
    def __init__(self, input_size: int, output_size: int, links: np.ndarray, weights: np.ndarray,
                 biases: np.ndarray, actFuns: [ActFun]):
        self.input_size = input_size
        self.output_size = output_size
        self.links = links.copy()
        self.weights = weights.copy()
        self.bias = biases.copy()
        self.inp = np.zeros(biases.shape)
        self.act = np.zeros(biases.shape)
        self.actFuns = []
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())

    def run(self, input: np.ndarray):
        self.act[:self.input_size] = input

        for i in range(self.input_size, len(self.bias)):
            # prev = np.where(self.links[:, i] == 1)
            #
            # inputs = self.act[prev]
            # weights = self.weights[prev, i]
            # result = np.sum(np.multiply(inputs, weights.T)) + self.bias[i] #TODO jak tu użyć softmaxa w ogole?
            wei = self.weights[:, i].reshape(-1, 1)
            result = np.sum(np.multiply(self.act, wei)) + self.bias[i] #TODO jak tu użyć softmaxa w ogole?
            self.act[i] = self.actFuns[i].compute(result)[0]


        return self.act[len(self.bias) - self.output_size:]

    def analyse(self):
        links = self.links
        comp_order = []
        while True:
            col_sum = np.sum(links, axis=1)
            zeros_ind = np.where(col_sum == 0)[0]
            if len(zeros_ind) == 0:
                break
            comp_order.append(zeros_ind)
            links[zeros_ind, :] = 0


    def test(self, test_input: [np.ndarray], test_output: [np.ndarray]) -> [float, float, float, np.ndarray]:
        confusion_matrix = np.zeros((self.output_size, self.output_size))

        for i in range(len(test_output)):
            net_result = self.run(test_input[i])
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(test_output[i])
            confusion_matrix[corr_class, pred_class] += 1

        tot_sum = np.sum(confusion_matrix)
        diag_sum = np.sum(np.diag(confusion_matrix))

        accuracy = diag_sum / tot_sum

        return [accuracy, self.average_precision(confusion_matrix), self.average_recall(confusion_matrix), confusion_matrix]

    def copy(self):

        actFuns = []
        for i in range(len(self.actFuns)):
            f = self.actFuns[i]
            if f is None:
                actFuns.append(None)
            else:
                actFuns.append(f.copy())
        return LooseNetwork(input_size=self.input_size, output_size=self.output_size, links=self.links.copy(),
                            weights=self.weights.copy(), biases=self.bias.copy(), actFuns=actFuns)


    def average_precision(self, conf_matrix: np.ndarray) -> float:
        row_sums = np.sum(conf_matrix, axis=1)
        diag = np.diag(conf_matrix)

        class_prec = np.zeros(diag.shape)

        for i in range(len(class_prec)):
            if row_sums[i] > 0:
                class_prec[i] = diag[i] / row_sums[i]
            else:
                class_prec[i] = 0

        return np.average(class_prec)


    def average_recall(self, conf_matrix: np.ndarray) -> float:
        col_sums = np.sum(conf_matrix, axis=0)
        diag = np.diag(conf_matrix)

        class_recall = np.zeros(diag.shape)

        for i in range(len(class_recall)):
            if col_sums[i] > 0:
                class_recall[i] = diag[i] / col_sums[i]
            else:
                class_recall[i] = 0

        return np.average(class_recall)

    def to_string(self):
        return f"{np.sum(self.links)}"

