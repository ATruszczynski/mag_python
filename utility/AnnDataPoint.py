from ann_point.AnnPoint import *
import numpy as np

from neural_network.ChaosNet import *
from ann_point.Functions import *

#TODO touch should get removed
class AnnDataPoint(): #TODO test
    def __init__(self, point: ChaosNet):
        self.point = point #TODO copy?
        self.ff = 0.
        self.acc = 0.
        self.prec = 0.
        self.rec = 0.
        self.touch = 0.
        self.f1 = 0.

    def add_data(self, new_ff: float, new_conf_mat: np.ndarray):
        self.ff = self.new_average(self.ff, new_ff)
        self.acc = self.new_average(self.acc, accuracy(new_conf_mat))
        self.prec = self.new_average(self.prec, average_precision(new_conf_mat))
        self.rec = self.new_average(self.rec, average_recall(new_conf_mat))
        self.f1 = self.new_average(self.f1, average_f1_score(new_conf_mat))
        self.touch += 1

    def new_average(self, curr_val: float, new_val: float) -> float:
        curr_sum = curr_val * self.touch
        new_sum = curr_sum + new_val
        result = new_sum / (self.touch + 1)

        return result

    def get_eff(self): #TODO check if av of av if same as other av of av
        return mean([self.acc, self.prec, self.rec])

