from ann_point.Functions import *
from utility.Utility2 import assert_acts_same


class HyperparameterRange:
    def __init__(self, init_wei: (float, float), init_bia: (float, float), it: (int, int), hidden_count: (int, int),
                 actFuns: [ActFun], mut_radius: (float, float), depr: (float, float), multi: (float, float),
                 p_prob: (float, float), c_prob: (float, float), p_rad: (float, float), aggrFuns: [ActFun] = None):
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

        if aggrFuns is None:
            self.aggrFuns = aggrFuns
        else:
            self.aggrFuns = []
            for i in range(len(aggrFuns)):
                self.aggrFuns.append(aggrFuns[i].copy())


        self.min_mut_radius = mut_radius[0]
        self.max_mut_radius = mut_radius[1]
        self.min_depr = depr[0]
        self.max_depr = depr[1]
        self.min_multi = multi[0]
        self.max_multi = multi[1]
        self.min_p_prob = p_prob[0]
        self.max_p_prob = p_prob[1]
        self.min_c_prob = c_prob[0]
        self.max_c_prob = c_prob[1]
        self.min_p_rad = p_rad[0]
        self.max_p_rad = p_rad[1]

    def copy(self):
        return HyperparameterRange(init_wei=(self.min_init_wei, self.max_init_wei), init_bia=(self.min_init_bia, self.max_init_bia),
                                   it=(self.min_it, self.max_it), hidden_count=(self.min_hidden, self.max_hidden),
                                   actFuns=self.actFunSet, mut_radius=(self.min_mut_radius, self.max_mut_radius),
                                   depr=(self.min_depr, self.max_depr),
                                   multi=(self.min_multi, self.max_multi),
                                   p_prob=(self.min_p_prob, self.max_p_prob),
                                   c_prob=(self.min_c_prob, self.max_c_prob),
                                   p_rad=(self.min_p_rad, self.max_p_rad), aggrFuns=self.aggrFuns)

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
    assert hrange1.min_depr == hrange2.min_depr
    assert hrange1.max_depr == hrange2.max_depr
    assert hrange1.min_multi == hrange2.min_multi
    assert hrange1.max_multi == hrange2.max_multi
    assert hrange1.min_p_prob == hrange2.min_p_prob
    assert hrange1.max_p_prob == hrange2.max_p_prob
    assert hrange1.min_c_prob == hrange2.min_c_prob
    assert hrange1.max_c_prob == hrange2.max_c_prob
    assert hrange1.min_p_rad == hrange2.min_p_rad
    assert hrange1.max_p_rad == hrange2.max_p_rad

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
