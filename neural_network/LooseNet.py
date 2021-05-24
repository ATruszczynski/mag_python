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
        self.neuron_count = len(self.bias)
        self.hidden_start_index = self.input_size
        self.hidden_end_index = self.neuron_count - self.output_size
        self.actFuns = []
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())

        self.comp_order = []

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

        # for comp in self.comp_order:
        #     wei = self.weights[:, comp]
        #     result = np.sum(np.multiply(self.act, wei), axis=0).reshape(-1, 1) + self.bias[comp]
        #     for i in range(len(comp)):
        #         c = comp[i]
        #         self.act[c] = self.actFuns[c].compute(np.array(result[i, 0]).reshape(1,1))


        return self.act[len(self.bias) - self.output_size:]

    def analyse(self):
        links = self.links.copy()
        self.comp_order = []
        links[list(range(0, self.input_size)), :] = 0
        untouched = np.array(list(range(self.input_size, len(self.bias))))
        while len(untouched) > 0:
            col_sum = np.sum(links[:, untouched], axis=0)
            zeros_ind = np.where(col_sum == 0)[0]
            ind_analysed = untouched[zeros_ind]
            self.comp_order.append(ind_analysed)
            links[ind_analysed, :] = 0
            untouched = np.array([u for u in untouched if u not in ind_analysed])


    def test(self, test_input: [np.ndarray], test_output: [np.ndarray]) -> [float, float, float, np.ndarray]:
        confusion_matrix = np.zeros((self.output_size, self.output_size))

        for i in range(len(test_output)):
            net_result = self.run(test_input[i])
            max = np.max(net_result)
            indi = np.where(net_result[:, 0] == max)[0]
            pred_class = indi[random.randint(0, len(indi) - 1)]
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
        return f"L:{np.round(np.sum(self.links), 0)}|D:{self.density()}|IO:{len(self.get_indices_of_no_input_neurons())}|OO:{len(self.get_indices_of_no_output_neurons())}|DC:{len(self.get_indices_of_disconnected_neurons())}"

    def get_col_indices(self):
        return list(range(self.input_size, len(self.bias)))

    def get_row_indices(self):
        result = []
        for i in self.get_col_indices():
            result.append(range(0, min(i, len(self.bias) - self.output_size + 1)))
        return result

    def get_indices_of_no_output_neurons(self):
        row_sum = np.sum(self.links, axis=1)
        zeros = list(np.where(row_sum[:self.hidden_end_index] == 0)[0])
        return zeros

    def get_indices_of_no_input_neurons(self):
        col_sum = np.sum(self.links, axis=0)
        zeros = list(np.where(col_sum == 0)[0])
        zeros = zeros[self.hidden_start_index:]
        return zeros

    def get_indices_of_disconnected_neurons(self):
        no_output = self.get_indices_of_no_output_neurons()
        no_input = self.get_indices_of_no_input_neurons()

        result = []
        for i in no_output:
            if i in no_input:
                result.append(i)
        return result

    def density(self):
        total = (self.neuron_count * self.neuron_count - self.neuron_count) / 2
        sum = np.sum(self.links)

        return round(sum / total, 3)


