from statistics import mean

from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff, Softmax
from neural_network.FeedForwardNeuralNetwork import *
import numpy as np


class FitnessFunction:
    def __init__(self):
        pass

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        pass

class CrossEffFitnessFunction(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        network = network_from_point(point[0], random.randint(0, 1000)) #TODO make sure this seed does nothing

        test_results = network.test(test_input=trainInputs, test_output=trainOutputs, lossFun=QuadDiff()) #TODO DONT USE TEST SETS IN TRAINING PROCESS WTF
        result = mean(test_results[0:3])

        cross = -test_results[4]

        return [(1 - result) ** 2 * cross, point[1]]

class CrossEffFitnessFunction2(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        network = network_from_point(point[0], random.randint(0, 1000)) #TODO make sure this seed does nothing

        test_results = network.test(test_input=trainInputs, test_output=trainOutputs, lossFun=QuadDiff()) #TODO DONT USE TEST SETS IN TRAINING PROCESS WTF
        result = mean(test_results[0:3])

        cross = -test_results[4]

        return [(1 - result) * cross, point[1]]



class CrossEffFitnessFunction3(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        network = network_from_point(point[0], random.randint(0, 1000)) #TODO make sure this seed does nothing
        out_size = network.neuronCounts[len(network.neuronCounts) - 1]
        confusion_matrix = np.zeros((out_size, out_size))

        result = 0
        losFun = CrossEntropy()
        sm = Softmax()

        for i in range(len(trainOutputs)):
            net_result = sm.compute(network.run(trainInputs[i]))
            pred_class = np.argmax(net_result)
            corr_class = np.argmax(trainOutputs[i])
            confusion_matrix[corr_class, pred_class] += 1
            result += losFun.compute(net_result, corr_class)

        result = -result
        eff = mean([accuracy(confusion_matrix), average_precision(confusion_matrix), average_recall(confusion_matrix)])

        return [(1 - eff) * result, point[1]]


class EffFitnessFunction(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        network = network_from_point(point[0], random.randint(0, 1000)) #TODO make sure this seed does nothing

        test_results = network.test(test_input=trainInputs, test_output=trainOutputs, lossFun=QuadDiff()) #TODO DONT USE TEST SETS IN TRAINING PROCESS WTF
        result = mean(test_results[0:3])

        return [result, point[1]]

