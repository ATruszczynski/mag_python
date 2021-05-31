import random

from ann_point.AnnPoint2 import AnnPoint2, CrossEntropy
from evolving_classifier.operators.MutationOperators import MutationOperator
import numpy as np

from neural_network.FeedForwardNeuralNetwork import network_from_point


class HillClimbOperator:
    def __init__(self, top: int, rep: int):
        self.top = top
        self.rep = rep
        pass

    def generate_hc_population(self, population: [AnnPoint2, float], radius: float):
        pass

class HillClimbMutationOperator(HillClimbOperator):
    def __init__(self, top: int, rep: int, mutationOperaotr: MutationOperator):
        super().__init__(top, rep)
        self.mo = mutationOperaotr

    def generate_hc_population(self, population: [AnnPoint2, float], radius: float):
        hill_climbed = []
        for hc in range(self.top):
            for p in range(self.rep):
                hill_climbed.append(self.mo.mutate(population[hc][0], 1, radius))

        return hill_climbed

# class HillClimbMomentumMutationOperator(HillClimbOperator):
#     def __init__(self, top: int, rep: int, mutationOperaotr: MutationOperator):
#         super().__init__(top, rep)
#         self.mo = mutationOperaotr
#         self.momentums = None
#
#     def generate_hc_population(self, population: [AnnPoint2, float], radius: float):
#         hill_climbed = []
#         for hc in range(self.top):
#             for p in range(self.rep):
#                 hill_climbed.append(self.mo.mutate(population[hc][0], 1, radius))
#
#         return hill_climbed



class HillClimbBackpropMutationOperator(HillClimbOperator):
    def __init__(self, top: int, rep: int, train_inputs: [np.ndarray], train_outputs: [np.ndarray]):
        super().__init__(top, rep)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs

    def generate_hc_population(self, population: [AnnPoint2, float], radius: float):
        hill_climbed = []

        for hc in range(self.top):
            point = population[hc][0]
            net = network_from_point(point, random.randint(0, 1001))
            net.lossFun = CrossEntropy()
            data = [[self.train_inputs[i], self.train_outputs[i]] for i in range(len(self.train_outputs))]
            wei_grads, bia_grads = net.get_grad(data)
            hill_climbed.append(point)
            for i in range(1, len(wei_grads)):
                wei = wei_grads[i]
                maxw = np.max(np.abs(wei))
                wei /= np.abs(maxw)
                bia = bia_grads[i]
                maxb = np.max(np.abs(bia))
                bia /= np.abs(maxb)
                wei_grads[i] = wei
                bia_grads[i] = bia
            for p in range(1, self.rep):
                step = random.uniform(0, radius)
                new = point.copy()
                for i in range(0, len(new.weights)):
                    new.weights[i] -= step * wei_grads[i + 1]
                    new.biases[i] -= step * bia_grads[i + 1]

                hill_climbed.append(new)

        return hill_climbed