import random

# from ann_point.AnnPoint2 import AnnPoint2
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
