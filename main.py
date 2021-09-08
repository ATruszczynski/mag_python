from ann_point.Functions import *
from neural_network.ChaosNet import ChaosNet
import pandas as pd
import os

# hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 10), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
#                              sqr_mut_prob=(0.05, 0.1), lin_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
#                              dstr_mut_prob=(0, 0), act_mut_prob=(0, 1))
#
# link1 = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
#                   [0, 0, 1, 0, 0, 0, 1, 0],
#                   [0, 1, 0, 1, 1, 1, 1, 1],
#                   [0, 1, 0, 0, 0, 1, 0, 0],
#                   [0, 1, 1, 1, 0, 0, 1, 1],
#                   [0, 0, 1, 1, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0]])
# wei1 = np.array([[0, 2, 0, 2, 0, 0, 0, 0],
#                  [0, 0, 2, 0, 0, 0, 2, 0],
#                  [0, 2, 0, 2, 2, 2, 2, 2],
#                  [0, 2, 0, 0, 0, 2, 0, 0],
#                  [0, 2, 2, 2, 0, 0, 2, 2],
#                  [0, 0, 2, 2, 0, 0, 0, 2],
#                  [0, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0]])
# bia1 = np.array([[0., -2, -3, -4, -5, -6, -7, -8]])
# actFuns1 = [None, ReLu(), ReLu(), ReLu(), ReLu(), ReLu(), None, None]
#
# link2 = np.array([[0, 0, 0, 1, 1, 0, 0, 0],
#                   [0, 0, 0, 1, 0, 1, 1, 0],
#                   [0, 1, 0, 1, 1, 0, 1, 1],
#                   [0, 1, 0, 0, 0, 1, 0, 0],
#                   [0, 1, 0, 1, 0, 0, 1, 1],
#                   [0, 1, 1, 1, 0, 0, 0, 1],
#                   [0, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, 0]])
# wei2 = np.array([[0, 0, 0, 2, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 0, 5, 5, 0],
#                  [0, 9, 0, 3, 7, 0, 2, 5],
#                  [0, 2, 0, 0, 0, 3, 0, 0],
#                  [0, 4, 0, 1, 0, 0, 7, 2],
#                  [0, 6, 7, 2, 0, 0, 0, 2],
#                  [0, 0, 0, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 0, 0]])
# bia2 = np.array([[0, -20, -30, -40, -50, -60, -70, -80]])
# actFuns2 = [None, TanH(), TanH(), TanH(), TanH(), TanH(), None, None]
#
# cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
#                aggrFun=GaussAct(), net_it=10, mutation_radius=-10, sqr_mut_prob=-20, lin_mut_prob=-30,
#                p_mutation_prob=-40, c_prob=-50, dstr_mut_prob=-60, act_mut_prob=-70)
#
# random.seed(1001)
# co = FinalCO4(hrange)
#
# c, d = co.crossover(cn1, cn2)
#
# ori = 1
from suites.ts_wines import test_suite_for_wine, get_data

data_frame = pd.read_csv(fr"data_sets{os.path.sep}winequality-white.csv")

print(data_frame.groupby("quality").count())