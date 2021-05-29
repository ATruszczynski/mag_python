from statistics import mean

from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff
from neural_network.FeedForwardNeuralNetwork import network_from_point
import numpy as np


class FitnessFunction:
    def __init__(self):
        pass

    def compute(self, point: AnnPoint2, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> float:
        pass

class CrossEffFitnessFunction(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, point: AnnPoint2, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> float:
        network = network_from_point(point, 1001) #TODO make sure this seed does nothing

        test_results = network.test(test_input=trainInputs, test_output=trainOutputs, lossFun=QuadDiff()) #TODO DONT USE TEST SETS IN TRAINING PROCESS WTF
        result = mean(test_results[0:3])

        cross = -test_results[4]

        return (1 - result) ** 2 * cross

