from math import inf
from statistics import mean

import numpy as np

from ann_point.Functions import ActFun, LossFun


class ChaosNet:
    def __init__(self, input_size: int, output_size: int, links: np.ndarray, weights: np.ndarray, biases: np.ndarray, actFuns: [ActFun], aggrFun: ActFun, maxIt: int = 1):
        #TODO validation
        self.input_size = input_size
        self.output_size = output_size
        self.links = links.copy()
        self.weights = weights.copy()
        self.bias = biases.copy()
        self.inp = np.zeros(biases.shape)
        self.act = np.zeros(biases.shape)
        self.neuron_count = self.bias.shape[1]
        self.hidden_start_index = self.input_size
        self.hidden_end_index = self.neuron_count - self.output_size
        self.actFuns = []
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())
        self.aggrFun = aggrFun
        self.comp_order = []
        self.max_it = maxIt
        self.comp_count = np.zeros(biases.shape)
        self.comp_order = None

    def run(self, input: np.ndarray):
        self.act[0, :self.input_size] = input.reshape(1, -1)
        for r in range(self.max_it):
            for n in range(self. hidden_start_index, self.hidden_end_index):
                wei = np.multiply(self.weights[:, n].reshape(-1, 1), self.links[:, n].reshape(-1, 1))
                self.inp[0, n] = np.sum(np.multiply(self.act, wei.T)) + self.bias[0, n]
                self.act[0, n] = self.actFuns[n].compute(self.inp[0, n])

        self.inp[0, self.hidden_end_index:] = np.sum(np.multiply(self.act.T, self.weights[:, self.hidden_end_index:]), axis=0) + self.bias[0, self.hidden_end_index:]
        self.act[0, self.hidden_end_index:] = self.aggrFun.compute(self.inp[0, self.hidden_end_index:])

        return self.act[0, self.hidden_end_index:]



        # if self.comp_order is None:
        #     self.get_comp_order()
        #
        # self.act[0, :self.input_size] = input.reshape(1, -1)
        # for batch in self.comp_order:
        #     wei = np.multiply(self.weights[:, batch].reshape(-1, len(batch)), self.links[:, batch].reshape(-1, len(batch)))
        #     result = np.sum(np.multiply(self.act, wei.T)) + self.bias[0, batch]
        #     # if isinstance(result, float):
        #     #     result = np.array([[result]])
        #     for n in range(len(batch)):
        #         self.act[0, batch[n]] = self.actFuns[batch[n]].compute(result[n])


        # return self.aggrFun.compute(self.act[0, self.hidden_end_index:])




    def get_comp_order(self):
        self.comp_order = []
        computed = list(range(self.input_size))
        links = self.links.copy()
        links[computed, :] = 0

        while len(computed) < self.neuron_count:
            in_degrees = np.sum(links, axis=0)
            zero_degrees_ind = np.where(in_degrees == 0)[0]
            batch = []
            for zdi in zero_degrees_ind:
                if zdi not in computed:
                    batch.append(zdi)
                    computed.append(zdi)
                    links[zdi, :] = 0

            self.comp_order.append(batch)

    def test(self, test_input: [np.ndarray], test_output: [np.ndarray], lf: LossFun = None) -> [float, float, float, np.ndarray]:
        out_size = self.output_size
        confusion_matrix = np.zeros((out_size, out_size))

        resultt = 0
        # resultt = -inf

        for i in range(len(test_output)):
            ti = test_input[i]
            to = test_output[i]
            net_result = self.run(ti)
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(to)
            confusion_matrix[corr_class, pred_class] += 1
            if lf is not None:
                # resultt = max(resultt, lf.compute(net_result, to))
                resultt += lf.compute(net_result, to)

        return [accuracy(confusion_matrix), average_precision(confusion_matrix), average_recall(confusion_matrix), confusion_matrix, resultt]

    def size(self):
        return -666

    def copy(self):
        actFuns = []
        for i in range(len(self.actFuns)):
            if self.actFuns[i] is None:
                actFuns.append(None)
            else:
                actFuns.append(self.actFuns[i].copy())

        return ChaosNet(input_size=self.input_size, output_size=self.output_size, weights=self.weights.copy(),
                        links=self.links.copy(), biases=self.bias.copy(), actFuns=actFuns, aggrFun=self.aggrFun.copy())


    # def get_comp_order(self):
    #     self.comp_order = [list(range(self.neuron_count))]
    #     self.comp_count = np.zeros(self.bias.shape)
    #
    #     to_recalculate = np.zeros(self.bias.shape)
    #
    #     for i in range(1, self.max_it):
    #         prev = self.comp_order[i - 1]
    #         for p in prev:
    #             to_update = self.get_neurons_to_update([p])
    #             to_recalculate_local = []
    #             for tu in to_update:
    #                 if tu <= p:
    #                     to_recalculate_local.append(tu)
    #                 else:
    #                     break
    #
    #             to_recalculate[to_recalculate_local] = 1
    #             self.comp_order.append(np.where(to_recalculate == 1)[0])

    # def get_computable_division(self, links: np.ndarray) -> [[int]]:
    #     pass
    #
    # def get_neurons_to_update(self, links: np.ndarray, touched: [int]) -> [int]:
    #     to_update = np.zeros(self.bias.shape)
    #
    #     analysed = self.neuron_count * [0]
    #     for t in touched:
    #         following = self.get_neurons_to_update_rec(links, t, analysed)
    #         to_update[following] = 1
    #
    #     to_update_ind = np.where(to_update == 1)[0]
    #     to_update_ind = sorted(to_update_ind)
    #
    #     return to_update_ind
    #
    # def get_neurons_to_update_rec(self, links: np.ndarray, current: int, analysed: [int]) -> [int]:
    #     to_update = [current]
    #     analysed.append(current)
    #     outgoing = np.where(links[current, :] == 1)[0]
    #     for o in outgoing:
    #         if o not in analysed:
    #             to_update.extend(self.get_neurons_to_update_rec(links, o, analysed))
    #             analysed.append(o)
    #
    #     return to_update


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









