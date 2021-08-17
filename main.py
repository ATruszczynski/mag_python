import numpy as np

# from evolving_classifier.operators.CrossoverOperator import FinalCrossoverOperator

from evolving_classifier.operators.MutationOperators import FinalMutationOperator, cut_into_puzzles, get_weight_mask

# from neural_network import *
# from neural_network.FeedForwardNeuralNetwork import *
# from ann_point.Functions import *
# from sklearn.linear_model import LinearRegression

# from testtuple import TestTuple
# from utility.Utility import get_default_hrange

# tt = TestTuple(1, 2, 3, [], 4, get_default_hrange(), FinalMutationOperator, FinalCrossoverOperator, None)

# print(np.log(10))

nc = 15
ic = 2
oc = 3

matrix = np.zeros((nc, nc))
n = 1
for i in range(nc):
    for j in range(nc):
        matrix[i, j] = n
        n += 1

mask = get_weight_mask(ic, oc, nc)

matrix = np.multiply(matrix, mask)

print(matrix)

P1, P2, P3, P4, P5 = cut_into_puzzles(matrix=matrix, i=ic, o=oc, start=4, num=2, lim=2, left=True)

print(P1)
print(P2)
print(P3)
print(P4)
print(P5)