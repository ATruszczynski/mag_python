from math import inf
from statistics import mean

import numpy as np

from ann_point.Functions import *
from utility.Utility2 import *

#TODO - A - links nie są potrzebne lol
# TODO - B - clean up file

class ChaosNet:
    def __init__(self, input_size: int, output_size: int, links: np.ndarray, weights: np.ndarray, biases: np.ndarray,
                 actFuns: [ActFun], aggrFun: ActFun, maxit: int, mutation_radius: float, wb_mutation_prob: float,
                 s_mutation_prob: float, p_mutation_prob: float, c_prob: float, r_prob: float):
        #TODO - A - validation

        assert links.shape[0] == weights.shape[0]
        assert links.shape[1] == weights.shape[1]
        assert biases.shape[0] == 1
        assert biases.shape[1] == weights.shape[1]
        assert len(actFuns) == biases.shape[1]

        self.links = links.copy()
        self.weights = weights.copy()
        self.biases = biases.copy()
        self.inp = np.zeros(biases.shape)
        self.act = np.zeros(biases.shape)
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

        assert self.neuron_count == weights.shape[1]
        assert weights.shape[1] - input_size - output_size == self.hidden_count
        assert self.neuron_count == self.links.shape[0]
        assert self.neuron_count == self.links.shape[1]

        # self.comp_count = np.zeros(biases.shape)#TODO - B - useless?
        self.hidden_comp_order = None
        self.maxit = maxit

        if mutation_radius == 0:
            raise Exception()
        self.mutation_radius = mutation_radius
        self.wb_mutation_prob = wb_mutation_prob
        self.s_mutation_prob = s_mutation_prob
        self.p_mutation_prob = p_mutation_prob
        self.c_prob = c_prob
        self.r_prob = r_prob

    def run(self, inputs: np.ndarray) -> np.ndarray:
        if self.hidden_comp_order is None:
            self.get_comp_order()

        self.act = np.zeros((self.neuron_count, inputs.shape[1]))
        self.inp = np.zeros((self.neuron_count, inputs.shape[1]))

        self.act[:self.input_size, :] = inputs

        for i in range(self.maxit):
            for n in self.hidden_comp_order: #TODO - S - czy w komp order jest output neurons? chyba nie
                wei = self.weights[:, n].reshape(-1, 1)
                self.inp[n, :] = np.dot(wei.T, self.act) + self.biases[0, n]
                self.act[n, :] = self.actFuns[n].compute(self.inp[n, :])

        self.inp[self.hidden_end_index:, :] = np.dot(self.weights[:, self.hidden_end_index:].T, self.act) + self.biases[0, self.hidden_end_index:].reshape(-1, 1)
        self.act[self.hidden_end_index:, :] = self.aggrFun.compute(self.inp[self.hidden_end_index:, :])

        return self.act[self.hidden_end_index:]

    #TODO - S - identify which vertices don't need to be processed
    def get_comp_order(self):
        #TODO - S - jak to reaguje na wierzchołek ukryyt bez wejścia?
        self.hidden_comp_order = []

        touched = list(range(self.hidden_start_index))
        layers = [list(range(self.hidden_start_index))]

        hidden_links = self.links[:self.hidden_end_index, :self.hidden_end_index].copy()

        #TODO - S - wyznaczanie które wierzch mają stopień wejśćia jest chyba bez sensu
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

    #TODO - S - write some tests of test
    def test(self, test_input: [np.ndarray], test_output: [np.ndarray], lf: LossFun = None) -> [float, float, float, np.ndarray]:
        out_size = self.output_size
        confusion_matrix = np.zeros((out_size, out_size))

        resultt = 0

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
                resultt += lfcc

        return [confusion_matrix, resultt]

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

    def size(self):
        return -666

    def edge_count(self):
        return sum(self.links)

    #TODO - B - dużo z tych funkcji jest do wyrzucenia prawd.
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

    def get_indices_of_connected_neurons(self):#TODO - A - can this be faster?
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
                        s_mutation_prob=self.s_mutation_prob, p_mutation_prob=self.p_mutation_prob, c_prob=self.c_prob,
                        r_prob=self.r_prob)

    def edge_count(self):
        how_many = np.sum(self.links)
        return how_many

    def density(self):
        how_many = np.sum(self.links)
        maxi = np.sum(get_weight_mask(self.input_size, self.output_size, self.neuron_count))

        return how_many/maxi

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


    #TODO - A - coś by się przydało z tym zrobić
    def to_string(self):
        actFunsString = self.get_act_fun_string()

        result = ""
        result += f"{self.input_size}|{self.output_size}|{self.neuron_count}|{round(np.sum(self.links))}|{self.maxit}|" \
                  f"{actFunsString}|" + f"{self.aggrFun.to_string()}|" \
                  f"mr:{round(self.mutation_radius, 5)}|wb:{round(self.wb_mutation_prob, 5)}|s:{round(self.s_mutation_prob, 5)}" \
                  f"|p:{round(self.p_mutation_prob, 5)}|c:{round(self.c_prob, 5)}|r:{round(self.r_prob, 5)}"

        return result

    def net_to_file(self, fpath: str):
        file = open(fpath, "w")
        file.write(f"input_size: {self.input_size}\n")
        file.write(f"output_size: {self.output_size}\n")
        file.write(f"neuron_count: {self.neuron_count}\n")
        file.write(f"links: \n{self.links}\n")
        file.write(f"weights: \n{self.weights}\n")
        file.write(f"biases: \n{self.biases}\n")
        file.write(f"actFuns: \n{self.get_act_fun_string()}\n")
        file.write(f"aggrFun: \n{self.aggrFun.to_string()}\n")

        file.close()

#TODO - S - SAVE NET AT THE END!!!!!!!!!!!
#TODO - S - flush files








