from ann_point.Functions import *

#TODO - S - operator krzyżowania dwóch dużych sieci nigdy nie zrobi mniejszej sieci :/
#TODO - S - zmiana linków powinna być raczej podyktowana prawd wb, nie prwad strukt; zmiana biasów powinna być podyktowana przez ps nie przez pwb

class HyperparameterRange:
    def __init__(self, init_wei: (float, float), init_bia: (float, float), it: (int, int), hidden_count: (int, int),
                 actFuns: [ActFun], mut_radius: (float, float), wb_mut_prob: (float, float), s_mut_prob: (float, float),
                 p_mutation_prob: (float, float), c_prob: (float, float), r_prob: (float, float)):
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
        self.min_p_mut_prob = p_mutation_prob[0]
        self.max_p_mut_prob = p_mutation_prob[1]
        self.min_c_prob = c_prob[0]
        self.max_c_prob = c_prob[1]
        self.min_r_prob = r_prob[0]
        self.max_r_prob = r_prob[1]
