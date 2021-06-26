from math import inf
from statistics import mean

import numpy as np

from ann_point.Functions import *


class ChaosNet:
    #TODO remove maxit as default
    def __init__(self, input_size: int, output_size: int, links: np.ndarray, weights: np.ndarray, biases: np.ndarray,
                 actFuns: [ActFun], aggrFun: ActFun, maxit: int, mutation_radius: float, wb_mutation_prob: float,
                 s_mutation_prob: float):
        #TODO validation

        assert links.shape[0] == weights.shape[0]
        assert links.shape[1] == weights.shape[1]
        assert biases.shape[0] == 1
        assert biases.shape[1] == weights.shape[1]
        assert len(actFuns) == biases.shape[1]

        self.links = links.copy()###---
        self.weights = weights.copy()###---
        self.biases = biases.copy()###---
        self.inp = np.zeros(biases.shape)
        self.act = np.zeros(biases.shape)
        self.actFuns = []###---
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())
        self.aggrFun = aggrFun###---

        #TODO make those private
        self.input_size = input_size###---
        self.output_size = output_size###---
        self.neuron_count = self.biases.shape[1]###---
        self.hidden_start_index = self.input_size###---
        self.hidden_end_index = self.neuron_count - self.output_size###---
        self.hidden_count = self.hidden_end_index - self.hidden_start_index###---

        assert self.neuron_count == weights.shape[1]
        assert weights.shape[1] - input_size - output_size == self.hidden_count
        assert self.neuron_count == self.links.shape[0]
        assert self.neuron_count == self.links.shape[1]

        self.comp_count = np.zeros(biases.shape)#TODO useless?
        self.hidden_comp_order = None###---
        self.maxit = maxit###---

        self.mutation_radius = mutation_radius###---
        self.wb_mutation_prob = wb_mutation_prob###---
        self.s_mutation_prob = s_mutation_prob###---

    # def run(self, input: np.ndarray, try_faster: bool = False):
    #     self.act[0, :self.input_size] = input.reshape(1, -1)
    #
    #     if self.hidden_comp_order is None:
    #         self.get_comp_order()
    #
    #     for i in range(self.maxit):
    #         for n in self.hidden_comp_order:
    #             # wei = np.multiply(self.weights[:, n].reshape(-1, 1), self.links[:, n].reshape(-1, 1)) #TODO co to robi?
    #             wei = self.weights[:, n].reshape(-1, 1)
    #             self.inp[0, n] = np.sum(np.multiply(self.act, wei.T)) + self.bias[0, n]
    #             self.act[0, n] = self.actFuns[n].compute(self.inp[0, n])
    #
    #     self.inp[0, self.hidden_end_index:] = np.sum(np.multiply(self.act.T, self.weights[:, self.hidden_end_index:]), axis=0) + self.bias[0, self.hidden_end_index:]
    #     self.act[0, self.hidden_end_index:] = self.aggrFun.compute(self.inp[0, self.hidden_end_index:])
    #
    #     return self.act[0, self.hidden_end_index:]

    def run(self, inputs: np.ndarray) -> np.ndarray:
        if self.hidden_comp_order is None:
            self.get_comp_order()

        self.act = np.zeros((self.neuron_count, inputs.shape[1]))
        self.inp = np.zeros((self.neuron_count, inputs.shape[1]))

        self.act[:self.input_size, :] = inputs

        for i in range(self.maxit):
            for n in self.hidden_comp_order:
                wei = self.weights[:, n].reshape(-1, 1)
                self.inp[n, :] = np.dot(wei.T, self.act) + self.biases[0, n]
                self.act[n, :] = self.actFuns[n].compute(self.inp[n, :])


        self.inp[self.hidden_end_index:, :] = np.dot(self.weights[:, self.hidden_end_index:].T, self.act) + self.biases[0, self.hidden_end_index:].reshape(-1, 1)
        self.act[self.hidden_end_index:, :] = self.aggrFun.compute(self.inp[self.hidden_end_index:, :])

        return self.act[self.hidden_end_index:]

        # for i in range(self.maxit):
        #     for n in self.hidden_comp_order:
        #         # wei = np.multiply(self.weights[:, n].reshape(-1, 1), self.links[:, n].reshape(-1, 1)) #TODO co to robi?
        #         wei = self.weights[:, n].reshape(-1, 1)
        #         self.inp[0, n] = np.sum(np.multiply(self.act, wei.T)) + self.bias[0, n]
        #         self.act[0, n] = self.actFuns[n].compute(self.inp[0, n])
        #
        # self.inp[0, self.hidden_end_index:] = np.sum(np.multiply(self.act.T, self.weights[:, self.hidden_end_index:]), axis=0) + self.bias[0, self.hidden_end_index:]
        # self.act[0, self.hidden_end_index:] = self.aggrFun.compute(self.inp[0, self.hidden_end_index:])

        # return self.act[self.hidden_end_index:]



    # def run_normal(self, input:np.ndarray):
    #     self.act[0, :self.input_size] = input.reshape(1, -1)
    #
    #     for n in range(self. hidden_start_index, self.hidden_end_index):
    #         wei = np.multiply(self.weights[:, n].reshape(-1, 1), self.links[:, n].reshape(-1, 1)) #TODO co to robi?
    #         self.inp[0, n] = np.sum(np.multiply(self.act, wei.T)) + self.bias[0, n]
    #         self.act[0, n] = self.actFuns[n].compute(self.inp[0, n])
    #
    #     self.inp[0, self.hidden_end_index:] = np.sum(np.multiply(self.act.T, self.weights[:, self.hidden_end_index:]), axis=0) + self.bias[0, self.hidden_end_index:]
    #     self.act[0, self.hidden_end_index:] = self.aggrFun.compute(self.inp[0, self.hidden_end_index:])
    #
    #     return self.act[0, self.hidden_end_index:]
    #
    # def run_faster(self, input: np.ndarray):
    #     if self.hidden_comp_order is None:
    #         self.get_comp_order()
    #
    #     self.act[0, :self.input_size] = input.reshape(1, -1)
    #     for batch in self.hidden_comp_order:
    #         wei = np.multiply(self.weights[:, batch].reshape(-1, len(batch)), self.links[:, batch].reshape(-1, len(batch)))
    #         self.inp[0, batch] = np.sum(np.multiply(self.act, wei.T), axis=1) + self.bias[0, batch]
    #         for n in range(len(batch)):
    #             self.act[0, batch[n]] = self.actFuns[batch[n]].compute(self.inp[0, batch[n]])
    #
    #     self.inp[0, self.hidden_end_index:] = np.sum(np.multiply(self.act.T, self.weights[:, self.hidden_end_index:]), axis=0) + self.bias[0, self.hidden_end_index:]
    #     self.act[0, self.hidden_end_index:] = self.aggrFun.compute(self.inp[0, self.hidden_end_index:])
    #
    #
    #     return self.act[0, self.hidden_end_index:]

    #TODO identify which vertices don't need to be processed
    def get_comp_order(self):
        # self.hidden_comp_order = []
        # computed = list(range(self.input_size))
        # computed.extend(range(self.hidden_end_index, self.neuron_count))
        # links = self.links.copy()
        # links[computed, :] = 0
        #
        # while len(computed) < self.neuron_count:
        #     in_degrees = np.sum(links, axis=0)
        #     zero_degrees_ind = np.where(in_degrees == 0)[0]
        #     batch = []
        #     for zdi in zero_degrees_ind:
        #         if zdi not in computed:
        #             batch.append(zdi)
        #             computed.append(zdi)
        #             links[zdi, :] = 0
        #
        #     self.hidden_comp_order.append(batch)

        # self.hidden_comp_order = []
        # hidden_links = self.links[self.hidden_start_index:self.hidden_end_index, self.hidden_start_index:self.hidden_end_index].copy()
        #
        # hidden_neuron_count = self.hidden_end_index - self.hidden_start_index
        # computed = []
        #
        # while len(computed) < hidden_neuron_count:
        #     in_degrees = np.sum(hidden_links, axis=0)
        #     zero_degrees_ind = np.where(in_degrees == 0)[0]
        #     batch = []
        #     for zdi in zero_degrees_ind:
        #         if zdi not in computed:
        #             batch.append(zdi)
        #             computed.append(zdi)
        #             hidden_links[zdi, :] = 0
        #
        #     self.hidden_comp_order.append(batch)
        #
        # for batch in self.hidden_comp_order:
        #     for i in range(len(batch)):
        #         batch[i] += self.input_size

        self.hidden_comp_order = []

        touched = list(range(self.hidden_start_index))
        layers = [list(range(self.hidden_start_index))]

        hidden_links = self.links[:self.hidden_end_index, :self.hidden_end_index].copy()

        while len(touched) < self.hidden_end_index:
            out_of_layer_edges = hidden_links[layers[-1]]
            oole_col_sums = np.sum(out_of_layer_edges, axis=0)
            out_of_layer_vertices = np.where(oole_col_sums > 0)[0].tolist()
            if len(out_of_layer_vertices) == 0:
                break
            new_layer = []
            for oolv in out_of_layer_vertices:
                if oolv not in touched:
                    new_layer.append(oolv)
                    touched.append(oolv)
            new_layer = sorted(new_layer)
            layers.append(new_layer)

        self.hidden_comp_order = [i for layer in layers[1:] for i in layer]








    def test(self, test_input: [np.ndarray], test_output: [np.ndarray], lf: LossFun = None) -> [float, float, float, np.ndarray]:
        out_size = self.output_size
        confusion_matrix = np.zeros((out_size, out_size))

        resultt = 0

        cont_inputs = np.hstack(test_input)
        net_results = self.run(cont_inputs)

        for i in range(len(test_output)):
            to = test_output[i]
            net_result = net_results[:, i]
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(to)
            confusion_matrix[corr_class, pred_class] += 1
            if lf is not None:
                resultt += lf.compute(net_result, to)

        return [accuracy(confusion_matrix), average_precision(confusion_matrix), average_recall(confusion_matrix), confusion_matrix, resultt]

    def set_internals(self, links: np.ndarray, weights: np.ndarray, biases: np.ndarray, actFuns: [ActFun], aggrFun: ActFun):
        self.links = links.copy()
        self.weights = weights.copy()
        self.biases = biases.copy()
        self.aggrFun = aggrFun.copy()

        self.actFuns = []
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())

        self.hidden_comp_order = None


    def size(self):
        return -666
    
    def get_indices_of_neurons_with_output(self):
        row_sum = np.sum(self.links, axis=1)
        ones = list(np.where(row_sum[:self.hidden_end_index] > 0)[0])
        ones.extend(list(range(self.hidden_end_index, self.neuron_count)))
        return ones

    def get_indices_of_neurons_with_input(self):
        col_sum = np.sum(self.links, axis=0)
        ones = list(np.where(col_sum > 0)[0])
        ones.extend(list(range(self.hidden_start_index)))
        return ones

    def get_indices_of_connected_neurons(self):#TODO can this be faster?
        connected_neurons = []
        with_input = self.get_indices_of_neurons_with_input()

        for i in with_input:
            connected_neurons.append(i)

        with_output = self.get_indices_of_neurons_with_output()
        for i in with_output:
            if i not in connected_neurons:
                connected_neurons.append(i)

        return connected_neurons


    def get_indices_of_used_neurons(self):
        no_output = self.get_indices_of_no_output_neurons()
        used = []

        for i in range(self.neuron_count):
            if i not in no_output:
                used.append(i)

        return used

    def get_number_of_used_neurons(self):
        return len(self.get_indices_of_used_neurons())

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

    def copy(self):#TODO test?
        actFuns = []
        for i in range(len(self.actFuns)):
            if self.actFuns[i] is None:
                actFuns.append(None)
            else:
                actFuns.append(self.actFuns[i].copy())

        return ChaosNet(input_size=self.input_size, output_size=self.output_size, weights=self.weights.copy(),
                        links=self.links.copy(), biases=self.biases.copy(), actFuns=actFuns, aggrFun=self.aggrFun.copy()
                        , maxit=self.maxit, mutation_radius=self.mutation_radius, wb_mutation_prob=self.wb_mutation_prob,
                        s_mutation_prob=self.s_mutation_prob)

    def calculate_distance_from_input(self):
        touched = []
        layers = []

        hidden_links = self.links


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









