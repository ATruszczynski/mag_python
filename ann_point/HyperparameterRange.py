from ann_point.Functions import *
from utility.Utility2 import assert_acts_same


class HyperparameterRange:
    def __init__(self, init_wei: (float, float), init_bia: (float, float), it: (int, int), hidden_count: (int, int),
                 actFuns: [ActFun], mut_radius: (float, float), sqr_mut_prob: (float, float), lin_mut_prob: (float, float),
                 p_mutation_prob: (float, float), c_prob: (float, float), dstr_mut_prob: (float, float)):
        self.min_init_wei = init_wei[0]
        self.max_init_wei = init_wei[1]
        self.min_init_bia = init_bia[0]
        self.max_init_bia = init_bia[1]
        self.min_it = it[0]
        self.max_it = it[1]
        self.min_hidden = hidden_count[0]
        self.max_hidden = hidden_count[1]

        self.actFunSet = []
        for i in range(len(actFuns)):
            self.actFunSet.append(actFuns[i].copy())

        self.min_mut_radius = mut_radius[0]
        self.max_mut_radius = mut_radius[1]
        self.min_sqr_mut_prob = sqr_mut_prob[0]
        self.max_sqr_mut_prob = sqr_mut_prob[1]
        self.min_lin_mut_prob = lin_mut_prob[0]
        self.max_lin_mut_prob = lin_mut_prob[1]
        self.min_p_mut_prob = p_mutation_prob[0]
        self.max_p_mut_prob = p_mutation_prob[1]
        self.min_c_prob = c_prob[0]
        self.max_c_prob = c_prob[1]
        self.min_dstr_mut_prob = dstr_mut_prob[0]
        self.max_dstr_mut_prob = dstr_mut_prob[1]

    def copy(self):
        return HyperparameterRange(init_wei=(self.min_init_wei, self.max_init_wei), init_bia=(self.min_init_bia, self.max_init_bia),
                                   it=(self.min_it, self.max_it), hidden_count=(self.min_hidden, self.max_hidden),
                                   actFuns=self.actFunSet, mut_radius=(self.min_mut_radius, self.max_mut_radius),
                                   sqr_mut_prob=(self.min_sqr_mut_prob, self.max_sqr_mut_prob),
                                   lin_mut_prob=(self.min_lin_mut_prob, self.max_lin_mut_prob),
                                   p_mutation_prob=(self.min_p_mut_prob, self.max_p_mut_prob),
                                   c_prob=(self.min_c_prob, self.max_c_prob),
                                   dstr_mut_prob=(self.min_dstr_mut_prob, self.max_dstr_mut_prob))

def assert_hranges_same(hrange1: HyperparameterRange, hrange2: HyperparameterRange):
    assert hrange1.min_init_wei == hrange2.min_init_wei
    assert hrange1.max_init_wei == hrange2.max_init_wei
    assert hrange1.min_init_bia == hrange2.min_init_bia
    assert hrange1.max_init_bia == hrange2.max_init_bia
    assert hrange1.min_it == hrange2.min_it
    assert hrange1.max_it == hrange2.max_it
    assert hrange1.min_hidden == hrange2.min_hidden
    assert hrange1.max_hidden == hrange2.max_hidden

    assert_acts_same(hrange1.actFunSet, hrange2.actFunSet)

    assert hrange1.min_mut_radius == hrange2.min_mut_radius
    assert hrange1.max_mut_radius == hrange2.max_mut_radius
    assert hrange1.min_sqr_mut_prob == hrange2.min_sqr_mut_prob
    assert hrange1.max_sqr_mut_prob == hrange2.max_sqr_mut_prob
    assert hrange1.min_lin_mut_prob == hrange2.min_lin_mut_prob
    assert hrange1.max_lin_mut_prob == hrange2.max_lin_mut_prob
    assert hrange1.min_p_mut_prob == hrange2.min_p_mut_prob
    assert hrange1.max_p_mut_prob == hrange2.max_p_mut_prob
    assert hrange1.min_c_prob == hrange2.min_c_prob
    assert hrange1.max_c_prob == hrange2.max_c_prob
    assert hrange1.min_dstr_mut_prob == hrange2.min_dstr_mut_prob
    assert hrange1.max_dstr_mut_prob == hrange2.max_dstr_mut_prob
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
