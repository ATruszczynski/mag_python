from math import inf, log10, sqrt
from statistics import mean, median
from typing import Any
from warnings import warn

import numpy as np

from ann_point.Functions import *
from utility.Utility2 import *

# TODO - B - clean up file

class ChaosNet:
    def __init__(self, input_size: int, output_size: int, links: np.ndarray, weights: np.ndarray, biases: np.ndarray,
                 actFuns: [ActFun], aggrFun: ActFun, net_it: int, mutation_radius: float, sqr_mut_prob: float,
                 lin_mut_prob: float, p_mutation_prob: float, c_prob: float, dstr_mut_prob: float):

        check_cond_in_cn_const(links.shape[0] == links.shape[1])
        check_cond_in_cn_const(links.shape[0] >= input_size + output_size)
        check_cond_in_cn_const(input_size >= 1)
        check_cond_in_cn_const(output_size >= 1)
        check_cond_in_cn_const(links.shape[0] == weights.shape[0])
        check_cond_in_cn_const(links.shape[1] == weights.shape[1])
        check_cond_in_cn_const(biases.shape[0] == 1)
        check_cond_in_cn_const(biases.shape[1] == weights.shape[1])
        check_cond_in_cn_const(len(actFuns) == biases.shape[1])
        check_cond_in_cn_const(net_it >= 1)
        # check_cond_in_cn_const(mutation_radius <= 0)
        check_cond_in_cn_const(sqr_mut_prob <= 0)
        check_cond_in_cn_const(lin_mut_prob <= 0)
        check_cond_in_cn_const(p_mutation_prob <= 0)
        check_cond_in_cn_const(c_prob <= 0)
        check_cond_in_cn_const(dstr_mut_prob <= 0)

        nonzl = np.where(links != 0)
        nonzw = np.where(weights != 0)
        if len(nonzl[0]) != len(nonzw[0]):
            warn("Edge with weight 0 present")
        else:
            for i in range(len(nonzl[0])):
                check_cond_in_cn_const(nonzl[0][i] == nonzw[0][i])
                check_cond_in_cn_const(nonzl[1][i] == nonzw[1][i])

        self.links = links.copy()
        self.weights = weights.copy()
        self.biases = biases.copy()
        self.inp = np.zeros((0, 0))
        self.act = np.zeros((0, 0))
        self.actFuns = []
        for i in range(len(actFuns)):
            if actFuns[i] is None:
                self.actFuns.append(None)
            else:
                self.actFuns.append(actFuns[i].copy())
        self.aggrFun = aggrFun

        #TODO - C - make those private
        self.input_size = input_size
        self.output_size = output_size
        self.neuron_count = self.biases.shape[1]
        self.hidden_start_index = self.input_size
        self.hidden_end_index = self.neuron_count - self.output_size
        self.hidden_count = self.hidden_end_index - self.hidden_start_index

        check_cond_in_cn_const(np.min(get_weight_mask(input_size, output_size, self.neuron_count) - self.links) >= 0)
        check_cond_in_cn_const(self.neuron_count == weights.shape[1])
        check_cond_in_cn_const(weights.shape[1] - input_size - output_size == self.hidden_count)
        check_cond_in_cn_const(self.neuron_count == self.links.shape[0])
        check_cond_in_cn_const(self.neuron_count == self.links.shape[1])

        self.hidden_comp_order = None
        self.net_it = net_it

        self.mutation_radius = mutation_radius

        self.p_mutation_prob = p_mutation_prob
        self.c_prob = c_prob

        self.sqr_mut_prob = log10(1. / self.neuron_count)
        self.lin_mut_prob = log10(1. / self.neuron_count)
        self.dstr_mut_prob = log10(1. / self.neuron_count**2)
        self.act_mut_prob = log10(1. / self.neuron_count**2)

        self.edge_count = np.sum(self.links)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        if self.hidden_comp_order is None:
            self.compute_comp_order()

        self.act = np.zeros((self.neuron_count, inputs.shape[1]))
        self.inp = np.zeros((self.neuron_count, inputs.shape[1]))

        self.act[:self.input_size, :] = inputs

        for i in range(self.net_it):
            for n in self.hidden_comp_order:
                wei = self.weights[:, n].reshape(-1, 1)
                self.inp[n, :] = np.dot(wei.T, self.act) + self.biases[0, n]
                self.act[n, :] = self.actFuns[n].compute(self.inp[n, :])

        self.inp[self.hidden_end_index:, :] = np.dot(self.weights[:, self.hidden_end_index:].T, self.act) + self.biases[0, self.hidden_end_index:].reshape(-1, 1)
        self.act[self.hidden_end_index:, :] = self.aggrFun.compute(self.inp[self.hidden_end_index:, :])
        # self.act[self.hidden_end_index:, :] = self.inp[self.hidden_end_index:, :]

        return self.act[self.hidden_end_index:]

    # def run(self, inputs: np.ndarray) -> np.ndarray:
    #     # if self.hidden_comp_order is None:
    #     #     self.compute_comp_order()
    #
    #     self.act = np.zeros((self.neuron_count, inputs.shape[1]))
    #     self.inp = np.zeros((self.neuron_count, inputs.shape[1]))
    #
    #     self.act[:self.input_size, :] = inputs
    #
    #     for i in range(self.net_it + 1):
    #         self.inp = np.dot(self.weights.T, self.act) + self.biases.T
    #         self.act = self.inp
    #         for j in range(self.hidden_start_index, self.hidden_end_index):
    #             self.act[j, :] = self.actFuns[j].compute(self.act[j, :])
    #
    #     self.act[self.hidden_end_index:, :] = self.aggrFun.compute(self.inp[self.hidden_end_index:, :])
    #
    #     return self.act[self.hidden_end_index:]

    def compute_comp_order(self):
        self.hidden_comp_order = []

        touched = list(range(self.hidden_start_index))
        layers = [list(range(self.hidden_start_index))]

        hidden_links = self.links[:self.hidden_end_index, :self.hidden_end_index].copy()

        while len(touched) < self.hidden_end_index:
            out_of_layer_edges = hidden_links[layers[-1], :]
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

    def test(self, test_input: [np.ndarray], test_output: [np.ndarray], lf: LossFun = None) -> [np.ndarray, float]:
        out_size = self.output_size
        confusion_matrix = np.zeros((out_size, out_size))

        resultt = []

        cont_inputs = np.hstack(test_input)
        net_results = self.run(cont_inputs)

        for i in range(len(test_output)):
            to = test_output[i]
            net_result = net_results[:, i].reshape(-1, 1)
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(to)
            confusion_matrix[corr_class, pred_class] += 1
            if lf is not None:
                lfcc = lf.compute(net_result, to)
                resultt.append(lfcc)

        test_res = [confusion_matrix]
        if lf is not None:
            test_res.append(mean(resultt))
        return test_res

    # def set_internals(self, links: np.ndarray, weights: np.ndarray, biases: np.ndarray, actFuns: [ActFun], aggrFun: ActFun):
    #     self.links = links.copy()
    #     self.weights = weights.copy()
    #     self.biases = biases.copy()
    #     self.aggrFun = aggrFun.copy()
    #
    #     self.actFuns = []
    #     for i in range(len(actFuns)):
    #         if actFuns[i] is None:
    #             self.actFuns.append(None)
    #         else:
    #             self.actFuns.append(actFuns[i].copy())
    #
    #     self.hidden_comp_order = None

    # def size(self):
    #     return -666

    #TODO - B - dużo z tych funkcji jest do wyrzucenia prawd.

    # def get_indices_of_neurons_with_output(self):
    #     row_sum = np.sum(self.links, axis=1)
    #     ones = list(np.where(row_sum[:self.hidden_end_index] > 0)[0])
    #     ones.extend(list(range(self.hidden_end_index, self.neuron_count)))
    #     return ones
    #
    # def get_indices_of_neurons_with_input(self):
    #     col_sum = np.sum(self.links, axis=0)
    #     ones = list(np.where(col_sum > 0)[0])
    #     ones.extend(list(range(self.hidden_start_index)))
    #     return ones

    # def get_indices_of_connected_neurons(self):#TODO - C - can this be faster?
    #     connected_neurons = []
    #     with_input = self.get_indices_of_neurons_with_input()
    #
    #     for i in with_input:
    #         connected_neurons.append(i)
    #
    #     with_output = self.get_indices_of_neurons_with_output()
    #     for i in with_output:
    #         if i not in connected_neurons:
    #             connected_neurons.append(i)
    #
    #     return connected_neurons

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
                        , net_it=self.net_it, mutation_radius=self.mutation_radius, sqr_mut_prob=self.sqr_mut_prob,
                        lin_mut_prob=self.lin_mut_prob, p_mutation_prob=self.p_mutation_prob, c_prob=self.c_prob,
                        dstr_mut_prob=self.dstr_mut_prob)

    def get_edge_count(self):
        how_many = np.sum(self.links)
        return how_many

    def get_max_edge_count(self):
        return np.sum(get_weight_mask(self.input_size, self.output_size, self.neuron_count))

    # def density(self):
    #     how_many = np.sum(self.links)
    #     maxi = np.sum(get_weight_mask(self.input_size, self.output_size, self.neuron_count))
    #
    #     return how_many/maxi

    def get_act_fun_string(self):
        actFunsString = ""
        for i in range(len(self.actFuns)):
            fun = self.actFuns[i]
            if fun is None:
                actFunsString += "-"
            else:
                actFunsString += self.actFuns[i].to_string()
            if i != len(self.actFuns) - 1:
                actFunsString += "|"

        return actFunsString

    def to_string(self):
        actFunsString = self.get_act_fun_string()

        result = ""
        result += f"{self.input_size}|{self.output_size}|{self.neuron_count}|{round(self.get_edge_count())}|{self.net_it}|" \
                  f"{actFunsString}|" + f"{self.aggrFun.to_string()}|" \
                  f"mr:{round(self.mutation_radius, 5)}|wb:{round(self.sqr_mut_prob, 5)}|s:{round(self.lin_mut_prob, 5)}" \
                  f"|p:{round(self.p_mutation_prob, 5)}|c:{round(self.c_prob, 5)}|r:{round(self.dstr_mut_prob, 5)}"

        return result











