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

# TODO - B - remove needless code from here

class FinalMutationOperator(MutationOperator):
    def __init__(self, hrange: HyperparameterRange):
        super().__init__(hrange)

    def mutate(self, point: ChaosNet) -> ChaosNet:
        point = point.copy()
        sqr_pm =    10 ** point.sqr_mut_prob
        lin_pm =    10 ** point.lin_mut_prob
        p_pm =      10 ** point.p_mutation_prob
        dstr_pm =  10 ** point.dstr_mut_prob
        act_pm = 10 ** point.act_mut_prob
        radius =    10 ** point.mutation_radius

        weights_shifted = gaussian_shift(point.weights, point.links, sqr_pm, radius)
        biases_shifted = gaussian_shift(point.biases, get_bias_mask(point.input_size, point.neuron_count), lin_pm, radius)

        nact = point.input_size * [None]
        for i in range(point.hidden_start_index, point.hidden_end_index):
            nact.append(conditional_try_choose_different(act_pm, point.actFuns[i], self.hrange.actFunSet))
        nact.extend(point.output_size * [None])

        aggr = conditional_try_choose_different(act_pm, point.aggrFun, self.hrange.actFunSet)

        links_rev, weights_rev = add_or_remove_edges(dstr_pm, point.links, weights_shifted, get_weight_mask(point.input_size, point.output_size, point.neuron_count), hrange=self.hrange)

        net_it = conditional_try_choose_different(lin_pm, point.net_it, list(range(self.hrange.min_it, self.hrange.max_it + 1)))

        rad_frac = 0.1
        mutation_radius = conditional_uniform_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
        sqr_mut_prob = conditional_uniform_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
        lin_mut_prob = conditional_uniform_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
        p_mutation_prob = conditional_uniform_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
        c_prob = conditional_uniform_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
        dstr_mut_prob = conditional_uniform_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
        act_mut_prob = conditional_uniform_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)

        mutation_radius = conditional_gaussian_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
        sqr_mut_prob = conditional_gaussian_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
        lin_mut_prob = conditional_gaussian_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
        p_mutation_prob = conditional_gaussian_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
        c_prob = conditional_gaussian_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
        dstr_mut_prob = conditional_gaussian_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
        act_mut_prob = conditional_gaussian_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)


        np = ChaosNet(input_size=point.input_size, output_size=point.output_size, links=links_rev, weights=weights_rev,
                       biases=biases_shifted, actFuns=nact, aggrFun=aggr, net_it=net_it, mutation_radius=mutation_radius, sqr_mut_prob=sqr_mut_prob,
                       lin_mut_prob=lin_mut_prob, p_mutation_prob=p_mutation_prob, c_prob=c_prob, dstr_mut_prob=dstr_mut_prob, act_mut_prob=act_mut_prob)

        return np
