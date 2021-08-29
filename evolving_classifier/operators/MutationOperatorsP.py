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

class FinalMutationOperatorP(MutationOperator):
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

        links = point.links
        weights = point.weights.copy()
        biases = point.biases.copy()
        nact = point.actFuns
        aggr = point.aggrFun
        net_it = point.net_it

        mutation_radius = point.mutation_radius
        sqr_mut_prob = point.sqr_mut_prob
        lin_mut_prob = point.lin_mut_prob
        p_mutation_prob = point.p_mutation_prob
        c_prob = point.c_prob
        dstr_mut_prob = point.dstr_mut_prob
        act_mut_prob = point.act_mut_prob

        p = random.random()

        if p <= p_pm:
            rad_frac = 0.1
            # mutation_radius = conditional_uniform_value_shift(1, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
            # sqr_mut_prob = conditional_uniform_value_shift(1, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
            # lin_mut_prob = conditional_uniform_value_shift(1, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
            # p_mutation_prob = conditional_uniform_value_shift(1, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
            # c_prob = conditional_uniform_value_shift(1, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
            # dstr_mut_prob = conditional_uniform_value_shift(1, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
            # act_mut_prob = conditional_uniform_value_shift(1, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)

            mutation_radius = conditional_gaussian_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
            sqr_mut_prob = conditional_gaussian_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
            lin_mut_prob = conditional_gaussian_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
            p_mutation_prob = conditional_gaussian_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
            c_prob = conditional_gaussian_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
            dstr_mut_prob = conditional_gaussian_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
            act_mut_prob = conditional_gaussian_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)

        else:
            rolls = choose_without_repetition(list(range(point.hidden_start_index, point.neuron_count)), max(1, ceil(0.2 * point.hidden_count)))

            # weights[:, rolls] = gaussian_shift(weights[:, rolls], point.links[:, rolls], sqr_pm, radius)
            weights[rolls, :] = gaussian_shift(weights[rolls, :], point.links[rolls, :], sqr_pm, radius)

            biases[:, rolls] = gaussian_shift(biases[:, rolls], get_bias_mask(point.input_size, point.neuron_count)[:, rolls], lin_pm, radius)

            links[:, rolls], weights[:, rolls] = add_or_remove_edges(dstr_pm, links[:, rolls], weights[:, rolls], point.input_size, point.output_size, get_weight_mask(point.input_size, point.output_size, point.neuron_count)[:, rolls], hrange=self.hrange)

            nact = point.input_size * [None]
            for i in range(point.hidden_start_index, point.hidden_end_index):
                if i in rolls:
                    nact.append(conditional_try_choose_different(act_pm, point.actFuns[i], self.hrange.actFunSet))
                else:
                    nact.append(point.actFuns[i].copy())
            nact.extend(point.output_size * [None])

            aggr = conditional_try_choose_different(act_pm, point.aggrFun, self.hrange.actFunSet)

            net_it = conditional_try_choose_different(lin_pm, point.net_it, list(range(self.hrange.min_it, self.hrange.max_it + 1)))


        # mutation_radius = conditional_gaussian_value_shift(p_pm, point.mutation_radius, self.hrange.min_mut_radius, self.hrange.max_mut_radius, rad_frac)
        # sqr_mut_prob = conditional_gaussian_value_shift(p_pm, point.sqr_mut_prob, self.hrange.min_sqr_mut_prob, self.hrange.max_sqr_mut_prob, rad_frac)
        # lin_mut_prob = conditional_gaussian_value_shift(p_pm, point.lin_mut_prob, self.hrange.min_lin_mut_prob, self.hrange.max_lin_mut_prob, rad_frac)
        # p_mutation_prob = conditional_gaussian_value_shift(p_pm, point.p_mutation_prob, self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob, rad_frac)
        # c_prob = conditional_gaussian_value_shift(p_pm, point.c_prob, self.hrange.min_c_prob, self.hrange.max_c_prob, rad_frac)
        # dstr_mut_prob = conditional_gaussian_value_shift(p_pm, point.dstr_mut_prob, self.hrange.min_dstr_mut_prob, self.hrange.max_dstr_mut_prob, rad_frac)
        # act_mut_prob = conditional_gaussian_value_shift(p_pm, point.act_mut_prob, self.hrange.min_act_mut_prob, self.hrange.max_act_mut_prob, rad_frac)


        np = ChaosNet(input_size=point.input_size, output_size=point.output_size, links=links, weights=weights,
                       biases=biases, actFuns=nact, aggrFun=aggr, net_it=net_it, mutation_radius=mutation_radius, sqr_mut_prob=sqr_mut_prob,
                       lin_mut_prob=lin_mut_prob, p_mutation_prob=p_mutation_prob, c_prob=c_prob, dstr_mut_prob=dstr_mut_prob, act_mut_prob=act_mut_prob)

        return np
