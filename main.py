import random

import numpy as np

# from evolving_classifier.operators.CrossoverOperator import FinalCrossoverOperator
from ann_point.Functions import *
from ann_point.HyperparameterRange import HyperparameterRange
from evolving_classifier.operators.CrossoverOperator import find_possible_cuts4, piece_together_from_puzzles, \
    FinalCrossoverOperator2
from evolving_classifier.operators.MutationOperators import FinalMutationOperator, cut_into_puzzles, get_weight_mask, \
    TanH

# from neural_network import *
# from neural_network.FeedForwardNeuralNetwork import *
# from ann_point.Functions import *
# from sklearn.linear_model import LinearRegression

# from testtuple import TestTuple
# from utility.Utility import get_default_hrange

# tt = TestTuple(1, 2, 3, [], 4, get_default_hrange(), FinalMutationOperator, FinalCrossoverOperator, None)

# print(np.log(10))

# nc = 15
# ic = 2
# oc = 3
#
# matrix = np.zeros((nc, nc))
# n = 1
# for i in range(nc):
#     for j in range(nc):
#         matrix[i, j] = n
#         n += 1
#
# mask = get_weight_mask(ic, oc, nc)
#
# matrix = np.multiply(matrix, mask)
#
# print(matrix)
#
# P1, P2, P3, P4, P5 = cut_into_puzzles(matrix=matrix, i=ic, o=oc, start=3, num=9, lim=2, left=False)
#
# print(P1)
# print(P2)
# print(P3)
# print(P4)
# print(P5)
from neural_network.ChaosNet import ChaosNet

hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 5), (0, 4), [ReLu(), Sigmoid(), GaussAct(), TanH()], mut_radius=(0, 1),
                             wb_mut_prob=(0.05, 0.1), s_mut_prob=(0.6, 0.7), p_mutation_prob=(0.4, 0.6), c_prob=(0.6, 0.6),
                             r_prob=(0, 0))

link1 = np.array([[0, 1, 0, 1],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
wei1 = np.array([[0., 1, 0, 4],
                 [0 , 2, 0, 5],
                 [0 , 0, 0, 0],
                 [0 , 0, 0, 0]])
bia1 = np.array([[0., -2, -3, -4]])
actFuns1 = [None, ReLu(), None, None]

link2 = np.array([[0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]])
wei2 = np.array([[0, 0, 0,  0,  0 ],
                 [0, 60, 10, 20, 0 ],
                 [0, 50, 100,  30, 40],
                 [0, 0, 0,  0,  0 ],
                 [0, 0, 0,  0,  0.]])
bia2 = np.array([[0, -20, -30, -40, -50]])
actFuns2 = [None, TanH(), TanH(), None, None]

cn1 = ChaosNet(input_size=1, output_size=2, weights=wei1, links=link1, biases=bia1, actFuns=actFuns1,
               aggrFun=SincAct(), maxit=2, mutation_radius=1, wb_mutation_prob=2, s_mutation_prob=3,
               p_mutation_prob=4, c_prob=5, r_prob=6)
cn2 = ChaosNet(input_size=1, output_size=2, weights=wei2, links=link2, biases=bia2, actFuns=actFuns2,
               aggrFun=GaussAct(), maxit=5, mutation_radius=10, wb_mutation_prob=20, s_mutation_prob=30,
               p_mutation_prob=40, c_prob=50, r_prob=60)

cuts = find_possible_cuts4(cn1, cn2, hrange)
print(cuts)

lp = cut_into_puzzles(matrix=cn1.weights, i=1, o=2, start=1, num=1, lim=2, left=True)
rp = cut_into_puzzles(matrix=cn2.weights, i=1, o=2, start=2, num=1, lim=1, left=False)

res = piece_together_from_puzzles(i=1, o=2, left_puzzles=lp, right_puzzles=rp)
# print(res)

# co = FinalCrossoverOperator2(hrange=hrange)
#
# random.seed(1001)
#
# cn3, cn4 = co.crossover(cn1, cn2)
#
# print(cn3.to_string())
# print(cn3.links)
# print(cn3.weights)
# print(cn3.biases)
# print(cn4.to_string())
# print(cn4.links)
# print(cn4.weights)
# print(cn4.biases)