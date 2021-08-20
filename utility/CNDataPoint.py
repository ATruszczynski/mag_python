import numpy as np

from neural_network.ChaosNet import *
from ann_point.Functions import *

class CNDataPoint(): #TODO - S - test
    def __init__(self, net: ChaosNet):
        self.net = net #TODO - B - copy?
        self.ff = 0.
        self.acc = 0.
        self.prec = 0.
        self.rec = 0.
        self.f1 = 0.

    def add_data(self, new_ff: float, new_conf_mat: np.ndarray):
        self.ff = new_ff
        self.acc = accuracy(new_conf_mat)
        self.prec = average_precision(new_conf_mat)
        self.rec = average_recall(new_conf_mat)
        self.f1 = average_f1_score(new_conf_mat)

    def get_eff(self):
        return mean([self.acc, self.prec, self.rec, self.f1])

    def copy(self):
        ncn = CNDataPoint(self.net.copy())
        ncn.ff = self.ff
        ncn.acc = self.acc
        ncn.prec = self.prec
        ncn.rec = self.rec
        ncn.f1 = self.f1

        return ncn

