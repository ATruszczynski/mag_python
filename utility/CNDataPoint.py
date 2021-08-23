import numpy as np

from neural_network.ChaosNet import *
from ann_point.Functions import *

#TODO - A - test
class CNDataPoint():
    def __init__(self, net: ChaosNet):
        self.net = net.copy()
        self.ff = 0.
        self.conf_mat = None

    def add_data(self, new_ff: float, new_conf_mat: np.ndarray):
        self.ff = new_ff
        self.conf_mat = new_conf_mat

    def get_eff(self):
        return efficiency(self.conf_mat)

    def get_acc(self):
        return accuracy(self.conf_mat)

    def get_avg_prec(self):
        return average_precision(self.conf_mat)

    def get_avg_rec(self):
        return average_recall(self.conf_mat)

    def get_avg_f1(self):
        return average_f1_score(self.conf_mat)

    def get_meff(self):
        return m_efficiency(self.conf_mat)

    def copy(self):
        ncn = CNDataPoint(self.net.copy())
        ncn.ff = self.ff
        ncn.conf_mat = self.conf_mat.copy()

        return ncn

