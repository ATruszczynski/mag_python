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
        sqr_pm =    10 ** point.depr
        modifier =    10 ** point.multi
        radius =    10 ** point.mutation_radius

        p_pm =   10 ** point.p_prob
        p_rad =  10 ** point.p_rad

        func_n = point.neuron_count - point.input_size

        sqr_pm = modifier / func_n
        dstr_pm = modifier / func_n**2

        weights_shifted = gaussian_shift(point.weights, point.links, sqr_pm, radius)
        biases_shifted = gaussian_shift(point.biases, get_bias_mask(point.input_size, point.neuron_count), sqr_pm, radius)
        nact = point.input_size * [None]
        for i in range(point.hidden_start_index, point.hidden_end_index):
            nact.append(conditional_try_choose_different(dstr_pm, point.actFuns[i], self.hrange.actFunSet))
        nact.extend(point.output_size * [None])

        if self.hrange.aggrFuns is None:
            aggr = conditional_try_choose_different(dstr_pm, point.aggrFun, self.hrange.actFunSet)
        else:
            aggr = conditional_try_choose_different(dstr_pm, point.aggrFun, self.hrange.aggrFuns)


        links_rev, weights_rev = add_or_remove_edges(dstr_pm, point.links, weights_shifted, get_weight_mask(point.input_size, point.output_size, point.neuron_count), hrange=self.hrange)

        minn = max(self.hrange.min_it, point.net_it - 1)
        maxn = min(self.hrange.max_it, point.net_it + 1)
        net_it = conditional_try_choose_different(dstr_pm, point.net_it, list(range(minn, maxn + 1)))

        mutation_radius = conditional_uniform_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, p_rad)
        sqr_mut_prob = conditional_uniform_value_shift(p_pm, point.depr, self.hrange.min_depr, self.hrange.max_depr, p_rad)
        lin_mut_prob = conditional_uniform_value_shift(p_pm, point.multi, self.hrange.min_multi, self.hrange.max_multi, p_rad)
        p_mutation_prob = conditional_uniform_value_shift(p_pm, point.p_prob, self.hrange.min_p_prob, self.hrange.max_p_prob, p_rad)
        c_prob = conditional_uniform_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, p_rad)
        dstr_mut_prob = conditional_uniform_value_shift(p_pm, point.p_rad, self.hrange.min_p_rad, self.hrange.max_p_rad, p_rad)
        # act_mut_prob = conditional_uniform_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)

        # mutation_radius = conditional_gaussian_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
        # sqr_mut_prob = conditional_gaussian_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
        # lin_mut_prob = conditional_gaussian_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
        # p_mutation_prob = conditional_gaussian_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
        # c_prob = conditional_gaussian_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
        # dstr_mut_prob = conditional_gaussian_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
        # act_mut_prob = conditional_gaussian_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)


        np = ChaosNet(input_size=point.input_size, output_size=point.output_size, links=links_rev, weights=weights_rev,
                      biases=biases_shifted, actFuns=nact, aggrFun=aggr, net_it=net_it, mutation_radius=mutation_radius, depr=sqr_mut_prob,
                      multi=lin_mut_prob, p_prob=p_mutation_prob, c_prob=c_prob, p_rad=dstr_mut_prob)

        return np
