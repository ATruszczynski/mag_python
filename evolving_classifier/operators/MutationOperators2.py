# import random
#
# from ann_point.HyperparameterRange import HyperparameterRange
# from unit_tests.utility_tests.OperatorUtility_test import gaussian_shift
# from utility.Mut_Utility import *
# from utility.Utility import *
# from utility.Utility2 import *
#
#
# class MutationOperator:
#     def __init__(self, hrange: HyperparameterRange):
#         self.hrange = hrange
#
#     def mutate(self, point: ChaosNet) -> ChaosNet:
#         pass
#
# # TODO - B - remove needless code from here
#
# class FinalMutationOperator(MutationOperator):
#     def __init__(self, hrange: HyperparameterRange):
#         super().__init__(hrange)
#
#     def mutate(self, point: ChaosNet) -> ChaosNet:
#         point = point.copy()
#         sqr_pm =    10 ** point.sqr_mut_prob
#         lin_pm =    10 ** point.lin_mut_prob
#         p_pm =      10 ** point.p_mutation_prob
#         dstr_pm =   10 ** point.dstr_mut_prob
#         radius =    10 ** point.mutation_radius
#
#         point.weights = gaussian_shift(point.weights, point.links, sqr_pm, radius)
#         point.biases = gaussian_shift(point.biases, get_bias_mask(point.input_size, point.neuron_count), lin_pm, radius)
#
#         for i in range(point.hidden_start_index, point.hidden_end_index):
#             point.actFuns[i] = conditional_try_choose_different(lin_pm, point.actFuns[i], self.hrange.actFunSet)
#
#         point.aggrFun = conditional_try_choose_different(lin_pm, point.aggrFun, self.hrange.actFunSet)
#
#         # TODO - C - this could prob just be a separate function
#         rad_frac = 0.1
#         spectrum = self.hrange.max_hidden - self.hrange.min_hidden
#         h_rad = max(1, round(spectrum * rad_frac))
#         minh = max(self.hrange.min_hidden, point.hidden_count - h_rad)
#         maxh = min(self.hrange.max_hidden, point.hidden_count + h_rad)
#         options = list(range(minh, maxh + 1))
#         point = change_neuron_count(point, self.hrange, conditional_try_choose_different(dstr_pm, point.hidden_count, options))
#
#         point.weights, point.links = add_or_remove_edges(dstr_pm, point.weights, point.links, get_weight_mask(point.input_size, point.output_size, point.neuron_count))
#
#         point.net_it = conditional_try_choose_different(lin_pm, point.net_it, list(range(self.hrange.min_it, self.hrange.max_it + 1)))
#
#         rad_frac = 0.1
#         point.mutation_radius = conditional_uniform_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
#
#         point.sqr_mut_prob = conditional_uniform_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
#
#         point.lin_mut_prob = conditional_uniform_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
#
#         point.p_mutation_prob = conditional_uniform_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
#
#         point.c_prob = conditional_uniform_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
#
#         point.dstr_mut_prob = conditional_uniform_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
#
#         return point
