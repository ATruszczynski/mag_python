import random

from ann_point.AnnPoint2 import AnnPoint2
from ann_point.HyperparameterRange import HyperparameterRange
from unit_tests.utility_tests.OperatorUtility_test import gaussian_shift
from utility.Mut_Utility import *
from utility.Utility import *


class MutationOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: ChaosNet) -> ChaosNet:
        pass

class FinalMutationOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: ChaosNet) -> ChaosNet:
        point = point.copy()
        wb_pm = point.wb_mutation_prob
        s_pm = point.s_mutation_prob
        p_pm = point.p_mutation_prob
        r_pm = point.r_prob
        radius = point.mutation_radius

        point.weights = gaussian_shift(point.weights, point.links, wb_pm, radius)

        #TODO tu można być coś zrobić żeby ignorowało zera
        # only_present = point.weights[np.where(point.links == 1)]
        # minW = 0
        # maxW = 0
        # if only_present.shape[0] != 0:
        #     minW = np.min(only_present)
        #     maxW = np.max(only_present)
        minW, maxW = get_min_max_values_of_matrix_with_mask(point.weights, point.links)
        point.weights = reroll_matrix(point.weights, point.links, r_pm, minW, maxW) # TODO zmień

        point.biases = gaussian_shift(point.biases, get_bias_mask(point.input_size, point.neuron_count), wb_pm, radius)

        #TODO tu można być coś zrobić żeby ignorowało zera
        only_present = point.biases[np.where(get_bias_mask(point.input_size, point.neuron_count) == 1)]
        point.biases = reroll_matrix(point.biases, get_bias_mask(point.input_size, point.neuron_count), r_pm, np.min(only_present), np.max(only_present)) # TODO zmień

        for i in range(point.hidden_start_index, point.hidden_end_index):
            point.actFuns[i] = conditional_try_choose_different(s_pm, point.actFuns[i], self.hrange.actFunSet)

        point.aggrFun = conditional_try_choose_different(s_pm, point.aggrFun, self.hrange.actFunSet)

        # TODO czy mutacja powinna umieć zmieniać liczbę neuronów?

        rad_frac = radius / self.hrange.max_mut_radius

        spectrum = self.hrange.max_hidden - self.hrange.min_hidden
        h_rad = max(1, round(spectrum * rad_frac))
        minh = max(self.hrange.min_hidden, point.hidden_count - h_rad) #TODO zrób coś z tym
        maxh = min(self.hrange.max_hidden, point.hidden_count + h_rad)
        options = list(range(minh, maxh + 1))
        point = change_neuron_count(point, self.hrange, conditional_try_choose_different(s_pm, point.hidden_count, options))

        # TODO zmienić to jakoś
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # probs = np.random.random(point.links.shape)
        # to_change = np.where(probs <= s_pm)
        # new_links = point.links.copy()
        # new_links[to_change] = 1 - new_links[to_change]
        # new_links = np.multiply(new_links, get_weight_mask(point.input_size, point.output_size, point.neuron_count))
        #
        # diffs = point.links - new_links
        # added_edges = np.where(diffs == -1)
        # minW = np.min(point.weights)
        # maxW = np.max(point.weights)
        # point.weights[added_edges] = np.random.uniform(minW, maxW, point.weights.shape)[added_edges]
        #
        # point.links = new_links
        #
        # point.weights = np.multiply(point.weights, point.links)
        point.weights, point.links = add_remove_weights(s_pm, point.weights, point.links, get_weight_mask(point.input_size, point.output_size, point.neuron_count))
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        point.maxit = conditional_try_choose_different(s_pm, point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        # point.mutation_radius = conditional_uniform_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
        #
        # point.wb_mutation_prob = conditional_uniform_value_shift(p_pm, point.wb_mutation_prob, self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob, rad_frac)
        #
        # point.s_mutation_prob = conditional_uniform_value_shift(p_pm, point.s_mutation_prob, self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob, rad_frac)
        #
        # point.p_mutation_prob = conditional_uniform_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
        #
        # point.c_prob = conditional_uniform_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
        #
        # point.r_prob = conditional_uniform_value_shift(p_pm, point.r_prob, self.hrange.min_r_prob, self.hrange.max_r_prob, rad_frac)




        point.mutation_radius = reroll_value(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius)

        point.wb_mutation_prob = reroll_value(p_pm, point.wb_mutation_prob, self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob)

        point.s_mutation_prob = reroll_value(p_pm, point.s_mutation_prob, self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob)

        point.p_mutation_prob = reroll_value(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob)

        point.c_prob = reroll_value(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob)

        point.r_prob = reroll_value(p_pm, point.r_prob, self.hrange.min_r_prob, self.hrange.max_r_prob)

        return point



# class SimpleCNMutation(MutationOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__(hrange)
#
#     def mutate(self, point: ChaosNet, wb_pm: float, s_pm: float, radius: float) -> ChaosNet:
#         probs = np.random.random(point.weights.shape)
#         change = np.zeros(point.weights.shape)
#         change[np.where(probs <= wb_pm)] = 1
#         wei_move = np.random.normal(0, radius, point.weights.shape)
#         wei_move = np.multiply(change, wei_move)
#         point.weights += wei_move
#         point.weights = np.multiply(point.weights, point.links)
#
#         probs = np.random.random((1, point.neuron_count))
#         change = np.zeros(probs.shape)
#         change[np.where(probs <= wb_pm)] = 1
#         bia_move = np.random.normal(0, radius, point.biases.shape)
#         bia_move = np.multiply(change, bia_move)
#         bia_move[0, :point.input_size] = 0
#         point.biases += bia_move
#
#         point.hidden_comp_order = None
#         if random.random() <= wb_pm:
#             point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))
#
#         return point

# class TestMutationOperator(MutationOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__(hrange)
#
#     def mutate(self, point: ChaosNet, wb_pm: float, s_pm: float, p_pm: float, radius: float) -> ChaosNet:
#         if radius <= 0:
#             probs = np.random.random(point.weights.shape)
#             to_replace = np.where(probs <= wb_pm)
#             wei_move = np.random.uniform(self.hrange.min_init_wei, self.hrange.max_init_wei, point.weights.shape)
#             point.weights[to_replace] = wei_move[to_replace]
#
#             probs = np.random.random(point.biases.shape)
#             bia_move = np.random.uniform(self.hrange.min_init_bia, self.hrange.max_init_bia, point.biases.shape)
#             to_replace = np.where(probs <= wb_pm)
#             point.biases[to_replace] = bia_move[to_replace]
#         else:
#             point.weights = gaussian_shift(point.weights, point.links, wb_pm, radius)
#
#             bias_mask = np.ones((1, point.neuron_count))
#             bias_mask[0, :point.input_size] = 0
#             point.biases = gaussian_shift(point.biases, bias_mask, wb_pm, radius)
#
#
#         for i in range(point.hidden_start_index, point.hidden_end_index):
#             if random.random() < s_pm:
#                 point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)
#
#         if random.random() < s_pm:
#             point.aggrFun = try_choose_different(point.aggrFun, self.hrange.actFunSet)
#
#         if random.random() < s_pm:
#             minh = max(self.hrange.min_hidden, point.hidden_count - 5)
#             maxh = min(self.hrange.max_hidden, point.hidden_count + 5)
#             options = list(range(minh, maxh + 1))
#             point = change_neuron_count(point, self.hrange, try_choose_different(point.hidden_count, options))
#
#         if random.random() <= s_pm: #TODO radius can be larger than 1
#             point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))
#
#         probs = np.random.random(point.links.shape)
#         to_change = np.where(probs <= s_pm)
#         new_links = point.links.copy()
#         new_links[to_change] = 1 - new_links[to_change]
#         new_links[:, :point.input_size] = 0
#         new_links[point.hidden_end_index:, :] = 0
#         np.fill_diagonal(new_links, 0)
#
#         diffs = point.links - new_links
#         added_edges = np.where(diffs == -1)
#         minW = np.min(point.weights)
#         maxW = np.max(point.weights)
#         point.weights[added_edges] = np.random.uniform(minW, maxW, point.weights.shape)[added_edges]
#
#         point.links = new_links
#
#         point.weights = np.multiply(point.weights, point.links)
#
#         if random.random() <= p_pm:
#             point.mutation_radius = random.uniform(self.hrange.min_mut_radius, self.hrange.max_mut_radius)
#
#         if random.random() <= p_pm:
#             point.wb_mutation_prob = random.uniform(self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob)
#
#         if random.random() <= p_pm:
#             point.s_mutation_prob = random.uniform(self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob)
#
#         if random.random() <= p_pm:
#             point.p_mutation_prob = random.uniform(self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob)
#
#         if random.random() <= p_pm:
#             point.c_prob = random.uniform(self.hrange.min_c_prob, self.hrange.max_c_prob)
#
#         net = ChaosNet(input_size=point.input_size, output_size=point.output_size, links=point.links, weights=point.weights,
#                        biases=point.biases, actFuns=point.actFuns, aggrFun=point.aggrFun, maxit=point.maxit, mutation_radius=point.mutation_radius,
#                        wb_mutation_prob=point.wb_mutation_prob, s_mutation_prob=point.s_mutation_prob, p_mutation_prob=point.p_mutation_prob,
#                        c_prob=point.c_prob)
#
#         net.weights = np.multiply(net.weights, get_weight_mask(net.input_size, net.output_size, net.neuron_count))
#         net.links = np.multiply(net.links, get_weight_mask(net.input_size, net.output_size, net.neuron_count))
#
#         return net.copy()
#
#
# class TestMutationOperatorGauss(MutationOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__(hrange)
#
#     def mutate(self, point: ChaosNet, wb_pm: float, s_pm: float, p_pm: float, radius: float) -> ChaosNet:
#         point.weights = gaussian_shift(point.weights, point.links, wb_pm, radius)
#
#         bias_mask = np.ones((1, point.neuron_count))
#         bias_mask[0, :point.input_size] = 0
#         point.biases = gaussian_shift(point.biases, bias_mask, wb_pm, radius)
#         if point.mutation_radius == 0:
#             raise Exception()
#
#         if random.random() <= p_pm:
#             rrr = point.mutation_radius
#             point.mutation_radius = random.uniform(self.hrange.min_mut_radius, self.hrange.max_mut_radius)
#             if point.mutation_radius == 0:
#                 raise Exception()
#
#         for i in range(point.hidden_start_index, point.hidden_end_index):
#             if random.random() < s_pm:
#                 point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)
#
#         if random.random() < s_pm:
#             point.aggrFun = try_choose_different(point.aggrFun, self.hrange.actFunSet)
#
#         if random.random() <= p_pm:
#             point.wb_mutation_prob = random.uniform(self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob)
#
#         if random.random() <= p_pm:
#             point.s_mutation_prob = random.uniform(self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob)
#
#         if random.random() <= p_pm:
#             point.p_mutation_prob = random.uniform(self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob)
#
#
#         return ChaosNet(input_size=point.input_size, output_size=point.output_size, links=point.links, weights=point.weights,
#                         biases=point.biases, actFuns=point.actFuns, aggrFun=point.aggrFun, maxit=point.maxit, mutation_radius=point.mutation_radius,
#                         wb_mutation_prob=point.wb_mutation_prob, s_mutation_prob=point.s_mutation_prob, p_mutation_prob=point.p_mutation_prob)


class SimpleAndStructuralCNMutation(MutationOperator):
    def __init__(self, hrange: HyperparameterRange, maxhjump: int):
        super().__init__(hrange)
        self.maxhjump = maxhjump

    def mutate(self, point: ChaosNet) -> ChaosNet:
        point = point.copy()
        wb_pm = point.wb_mutation_prob
        s_pm = point.s_mutation_prob
        p_pm = point.p_mutation_prob
        radius = point.mutation_radius

        # mask = get_mask(input_size=point.input_size, output_size=point.output_size, neuron_count=point.neuron_count)

        probs = np.random.random(point.weights.shape)
        change = np.zeros(point.weights.shape)
        change[np.where(probs <= wb_pm)] = 1
        wei_move = np.random.normal(0, radius, point.weights.shape)
        wei_move = np.multiply(change, wei_move)
        point.weights += wei_move
        point.weights = np.multiply(point.weights, point.links)

        probs = np.random.random((1, point.neuron_count))
        change = np.zeros(probs.shape)
        change[np.where(probs <= wb_pm)] = 1
        change[0, :point.input_size] = 0
        bia_move = np.random.normal(0, radius, point.biases.shape)
        bia_move = np.multiply(change, bia_move)
        point.biases += bia_move

        if random.random() <= s_pm: #TODO radius can be larger than 1
            point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        probs = np.random.random(point.links.shape)
        to_change = np.where(probs <= s_pm)
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
            if random.random() < s_pm:
                point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)

        if random.random() < s_pm:
            point.aggrFun = try_choose_different(point.aggrFun, self.hrange.actFunSet)

        if random.random() < s_pm:
            minh = max(self.hrange.min_hidden, point.hidden_count - self.maxhjump)
            maxh = min(self.hrange.max_hidden, point.hidden_count + self.maxhjump)
            options = list(range(minh, maxh + 1))
            point = change_neuron_count(point, self.hrange, try_choose_different(point.hidden_count, options))

        point.hidden_comp_order = None

        if random.random() < p_pm:
            min_rad = self.hrange.min_mut_radius
            max_rad = self.hrange.max_mut_radius
            point.mutation_radius = random.uniform(min_rad, max_rad)

        if random.random() < p_pm:
            min_rad = self.hrange.min_wb_mut_prob
            max_rad = self.hrange.max_wb_mut_prob
            point.wb_mutation_prob = random.uniform(min_rad, max_rad)

        if random.random() < p_pm:
            min_rad = self.hrange.min_s_mut_prob
            max_rad = self.hrange.max_s_mut_prob
            point.s_mutation_prob = random.uniform(min_rad, max_rad)

        if random.random() < p_pm:
            min_rad = self.hrange.min_p_mut_prob
            max_rad = self.hrange.max_p_mut_prob
            point.p_mutation_prob = random.uniform(min_rad, max_rad)

        # point = ChaosNet(point.input_size, point.output_size, links=point.links, weights=point.weights, biases=point.biases,
        #                  actFuns=point.actFuns, aggrFun=point.aggrFun, maxit=point.maxit)
        return point.copy()


#
# class SimpleAndStructuralCNMutation2(MutationOperator):
#     def __init__(self, hrange: HyperparameterRange, maxhjump: int):
#         super().__init__(hrange)
#         self.maxhjump = maxhjump
#
#     def mutate(self, point: ChaosNet, wb_pm: float, s_pm: float, p_pm: float, radius: float) -> ChaosNet:
#         point = point.copy()
#
#         probs = np.random.random(point.weights.shape)
#         # change = np.zeros(point.weights.shape)
#         # change[np.where(probs <= wb_pm)] = 1
#         # wei_move = np.random.normal(0, radius, point.weights.shape)
#         # wei_move = np.multiply(change, wei_move)
#         # point.weights += wei_move
#         # point.weights = np.multiply(point.weights, point.links)
#         to_change = np.where(probs <= wb_pm)
#         point.weights[to_change] = np.random.uniform(self.hrange.min_init_wei, self.hrange.max_init_wei, point.weights.shape)[to_change]
#
#         probs = np.random.random((1, point.neuron_count))
#         # change = np.zeros(probs.shape)
#         # change[np.where(probs <= wb_pm)] = 1
#         # change[0, :point.input_size] = 0
#         # bia_move = np.random.normal(0, radius, point.biases.shape)
#         # bia_move = np.multiply(change, bia_move)
#         # point.biases += bia_move
#         to_change = np.where(probs <= wb_pm)
#         point.biases[to_change] = np.random.uniform(self.hrange.min_init_wei, self.hrange.max_init_wei, point.biases.shape)[to_change]
#
#         if random.random() <= s_pm: #TODO radius can be larger than 1
#             point.maxit = try_choose_different(point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))
#
#         probs = np.random.random(point.links.shape)
#         to_change = np.where(probs <= s_pm)
#         new_links = point.links.copy()
#         new_links[to_change] = 1 - new_links[to_change]
#         new_links[:, :point.input_size] = 0
#         new_links[point.hidden_end_index:, :] = 0
#         np.fill_diagonal(new_links, 0)
#
#         diffs = point.links - new_links
#         added_edges = np.where(diffs == -1)
#         minW = np.min(point.weights)
#         maxW = np.max(point.weights)
#         point.weights[added_edges] = np.random.uniform(minW, maxW, point.weights.shape)[added_edges]
#
#         point.links = new_links
#
#         point.weights = np.multiply(point.weights, point.links)
#
#         for i in range(point.hidden_start_index, point.hidden_end_index):
#             if random.random() < s_pm:
#                 point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)
#
#         if random.random() < s_pm:
#             point.aggrFun = try_choose_different(point.aggrFun, self.hrange.actFunSet)
#
#         if random.random() < s_pm:
#             minh = max(self.hrange.min_hidden, point.hidden_count - self.maxhjump)
#             maxh = min(self.hrange.max_hidden, point.hidden_count + self.maxhjump)
#             options = list(range(minh, maxh + 1))
#             point = change_neuron_count(point, self.hrange, try_choose_different(point.hidden_count, options))
#
#         point.hidden_comp_order = None
#
#         if random.random() < p_pm:
#             min_rad = self.hrange.min_mut_radius
#             max_rad = self.hrange.max_mut_radius
#             point.mutation_radius = random.uniform(min_rad, max_rad)
#
#         if random.random() < p_pm:
#             min_rad = self.hrange.min_wb_mut_prob
#             max_rad = self.hrange.max_wb_mut_prob
#             point.wb_mutation_prob = random.uniform(min_rad, max_rad)
#
#         if random.random() < p_pm:
#             min_rad = self.hrange.min_s_mut_prob
#             max_rad = self.hrange.max_s_mut_prob
#             point.s_mutation_prob = random.uniform(min_rad, max_rad)
#
#         if random.random() < p_pm:
#             min_rad = self.hrange.min_p_mut_prob
#             max_rad = self.hrange.max_p_mut_prob
#             point.p_mutation_prob = random.uniform(min_rad, max_rad)
#
#         # point = ChaosNet(point.input_size, point.output_size, links=point.links, weights=point.weights, biases=point.biases,
#         #                  actFuns=point.actFuns, aggrFun=point.aggrFun, maxit=point.maxit)
#         return point.copy()

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
