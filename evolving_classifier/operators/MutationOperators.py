import random

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.HyperparameterRange import HyperparameterRange
from utility.Mut_Utility import *
from utility.Utility import *


class MutationOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: ChaosNet, pm: float, radius: float) -> ChaosNet:
        pass

class SimpleCNMutation(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: ChaosNet, pm: float, radius: float) -> ChaosNet:
        probs = np.random.random(point.weights.shape)
        change = np.zeros(point.weights.shape)
        change[np.where(probs <= pm)] = 1
        wei_move = np.random.normal(0, radius, point.weights.shape)
        wei_move = np.multiply(change, wei_move)
        point.weights += wei_move
        point.weights = np.multiply(point.weights, point.links)

        probs = np.random.random((1, point.neuron_count))
        change = np.zeros(probs.shape)
        change[np.where(probs <= pm)] = 1
        bia_move = np.random.normal(0, radius, point.biases.shape)
        bia_move = np.multiply(change, bia_move)
        point.biases += bia_move

        point.hidden_comp_order = None
        if random.random() <= pm:
            point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        return point


class SimpleAndStructuralCNMutation(MutationOperator):
    def __init__(self, hrange: HyperparameterRange, maxhjump: int):
        super().__init__(hrange)
        self.maxhjump = maxhjump

    def mutate(self, point: ChaosNet, pm: float, radius: float) -> ChaosNet:
        probs = np.random.random(point.weights.shape)
        change = np.zeros(point.weights.shape)
        change[np.where(probs <= pm)] = 1
        wei_move = np.random.normal(0, radius, point.weights.shape)
        wei_move = np.multiply(change, wei_move)
        point.weights += wei_move
        point.weights = np.multiply(point.weights, point.links)

        probs = np.random.random((1, point.neuron_count))
        change = np.zeros(probs.shape)
        change[np.where(probs <= pm)] = 1
        change[0, :point.input_size] = 0
        bia_move = np.random.normal(0, radius, point.biases.shape)
        bia_move = np.multiply(change, bia_move)
        point.biases += bia_move

        if random.random() <= pm * radius: #TODO radius can be larger than 1
            point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        probs = np.random.random(point.links.shape)
        to_change = np.where(probs <= pm)
        new_links = point.links.copy()
        new_links[to_change] = 1 - new_links[to_change]
        new_links[:, :point.input_size] = 0
        new_links[point.hidden_end_index:, :] = 0
        np.fill_diagonal(new_links, 0)

        diffs = point.links - new_links
        added_edges = np.where(diffs == -1)
        minW = np.min(point.weights)
        maxW = np.max(point.weights)
        point.weights[added_edges] = np.random.uniform(minW, maxW, point.weights.shape)[added_edges]

        point.links = new_links

        point.weights = np.multiply(point.weights, point.links)

        for i in range(point.hidden_start_index, point.hidden_end_index):
            if random.random() < pm * radius:
                point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)

        if random.random() < pm * radius:
            point.aggrFun = try_choose_different(point.aggrFun, self.hrange.actFunSet)

        if random.random() < pm * radius:
            minh = max(self.hrange.min_hidden, point.hidden_count - self.maxhjump)
            maxh = min(self.hrange.max_hidden, point.hidden_count + self.maxhjump)
            options = list(range(minh, maxh + 1))
            point = change_neuron_count(point, self.hrange, try_choose_different(point.hidden_count, options))

        point.hidden_comp_order = None
        return point

#
# class SimpleMutationOperator():
#     def __init__(self, hrange: HyperparameterRange):
#         self.hrange = hrange
#
#     def mutate(self, point: AnnPoint, pm: float, radius: float) -> AnnPoint:
#         point = point.copy()
#
#         if random.random() < pm * radius:
#             current = len(point.neuronCounts) - 2
#             minhl = max(current - 1, self.hrange.hiddenLayerCountMin)
#             maxhl = min(current + 1, self.hrange.hiddenLayerCountMax)
#
#             new_lay_count = try_choose_different(current, range(minhl, maxhl + 1))
#             diff = new_lay_count - current
#             if diff > 0:
#                 point = add_layers(point=point, howMany=diff, hrange=self.hrange)
#             elif diff < 0:
#                 point = remove_layers(point=point, howMany=-diff)
#
#         for i in range(1, len(point.neuronCounts) - 1):
#             if random.random() < pm:
#                 point.neuronCounts[i] = round(get_in_radius(point.neuronCounts[i], self.hrange.neuronCountMin, self.hrange.neuronCountMax, radius))
#
#         for i in range(len(point.actFuns)):
#             if random.random() < pm:
#                 point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)
#
#         if random.random() < pm * radius:
#             point.lossFun = try_choose_different(point.lossFun, self.hrange.lossFunSet)
#
#         if random.random() < pm:
#             point.learningRate = get_in_radius(point.learningRate, self.hrange.learningRateMin, self.hrange.learningRateMax, radius)
#
#         if random.random() < pm:
#             point.momCoeff = get_in_radius(point.momCoeff, self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax, radius)
#
#         if random.random() < pm:
#             point.batchSize = get_in_radius(point.batchSize, self.hrange.batchSizeMin, self.hrange.batchSizeMax, radius)
#
#         return point
