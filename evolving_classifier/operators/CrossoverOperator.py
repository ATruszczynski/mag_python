import random

from ann_point.AnnPoint import AnnPoint
from ann_point.AnnPoint2 import AnnPoint2, point_from_layer_tuples
from utility.Mut_Utility import *
from utility.Utility import get_Xu_matrix
import numpy as np


class CrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint]:
        pass

class SimpleCrossoverOperator:
    def __init__(self):
        pass

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint2) -> [AnnPoint2, AnnPoint]:
        pass


