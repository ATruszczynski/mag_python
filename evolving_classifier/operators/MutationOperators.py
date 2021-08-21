import random

from ann_point.HyperparameterRange import HyperparameterRange
from unit_tests.utility_tests.OperatorUtility_test import gaussian_shift
from utility.Mut_Utility import *
from utility.Utility import *
from utility.Utility2 import *


class MutationOperator:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange

    def mutate(self, point: ChaosNet) -> ChaosNet:
        pass
# TODO - S - czy poprawnie są zmienione prawd zmian kwadr i liniowych?
# TODO - S - does reroll make sense at all? It is included in large radius wb shifts mutations anyway?
# TODO - S - crossover of two identical networks is a waste of time?
# TODO - S - czy lepiej żeby efektywność używała minimów?
# TODO - B - remove needless code from here

class FinalMutationOperator(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: ChaosNet) -> ChaosNet:
        point = point.copy()
        wb_pm =  10 ** point.wb_mutation_prob
        s_pm =   10 ** point.s_mutation_prob
        p_pm =   10 ** point.p_mutation_prob
        r_pm =   10 ** point.r_prob
        radius = 10 ** point.mutation_radius

        point.weights = gaussian_shift(point.weights, point.links, wb_pm, radius)
        point.biases = gaussian_shift(point.biases, get_bias_mask(point.input_size, point.neuron_count), s_pm, radius)

        for i in range(point.hidden_start_index, point.hidden_end_index):
            point.actFuns[i] = conditional_try_choose_different(s_pm, point.actFuns[i], self.hrange.actFunSet)

        point.aggrFun = conditional_try_choose_different(s_pm, point.aggrFun, self.hrange.actFunSet)

        # TODO - S - czy mutacja powinna umieć zmieniać liczbę neuronów?
        # rad_frac = 0.1
        # spectrum = self.hrange.max_hidden - self.hrange.min_hidden
        # h_rad = max(1, round(spectrum * rad_frac))
        # minh = max(self.hrange.min_hidden, point.hidden_count - h_rad)
        # maxh = min(self.hrange.max_hidden, point.hidden_count + h_rad)
        # # maxh = point.hidden_count
        # options = list(range(minh, maxh + 1))
        # point = change_neuron_count(point, self.hrange, conditional_try_choose_different(s_pm, point.hidden_count, options))

        point.weights, point.links = add_remove_weights(r_pm, point.weights, point.links, get_weight_mask(point.input_size, point.output_size, point.neuron_count))
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        point.maxit = conditional_try_choose_different(s_pm, point.maxit, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        rad_frac = 0.1
        point.mutation_radius = conditional_uniform_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)

        point.wb_mutation_prob = conditional_uniform_value_shift(p_pm, point.wb_mutation_prob, self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob, rad_frac)

        point.s_mutation_prob = conditional_uniform_value_shift(p_pm, point.s_mutation_prob, self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob, rad_frac)

        point.p_mutation_prob = conditional_uniform_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)

        point.c_prob = conditional_uniform_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)

        point.r_prob = conditional_uniform_value_shift(p_pm, point.r_prob, self.hrange.min_r_prob, self.hrange.max_r_prob, rad_frac)

        return point
