from ann_point.Functions import *

class HyperparameterRange:
    def __init__(self, init_wei: (float, float), init_bia: (float, float), it: (int, int), hidden_count: (int, int),
                 actFuns: [ActFun], mut_radius: (float, float), wb_mut_prob: (float, float), s_mut_prob: (float, float)):
        self.min_init_wei = init_wei[0]
        self.max_init_wei = init_wei[1]
        self.min_init_bia = init_bia[0]
        self.max_init_bia = init_bia[1]
        self.min_it = it[0]
        self.max_it = it[1]
        self.min_hidden = hidden_count[0]
        self.max_hidden = hidden_count[1]

        self.actFunSet = actFuns

        self.min_mut_radius = mut_radius[0]
        self.max_mut_radius = mut_radius[1]
        self.min_wb_mut_prob = wb_mut_prob[0]
        self.max_wb_mut_prob = wb_mut_prob[1]
        self.min_s_mut_prob = s_mut_prob[0]
        self.max_s_mut_prob = s_mut_prob[1]
