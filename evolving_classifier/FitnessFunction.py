from statistics import mean

from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff, Softmax
from neural_network.FeedForwardNeuralNetwork import *
import numpy as np

from sklearn.linear_model import LinearRegression


class FitnessFunction:
    def __init__(self):
        pass

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        pass

class ProgressFF(FitnessFunction):
    def __init__(self, learningIts):
        super().__init__()
        self.learningIts = learningIts

    def compute(self, point: [AnnPoint2, int], trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, int]:
        network = network_from_point(point[0], random.randint(0, 1000))

        results = []

        for i in range(self.learningIts):
            network.train(inputs=trainInputs, outputs=trainOutputs, epochs=1) #TODO could use validation probably
            test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
            result = mean(test_results[0:3])
            results.append(result)

        y = np.array(results)
        x = np.array(list(range(0, self.learningIts)))

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        reg = LinearRegression().fit(x, y)
        slope = reg.coef_

        return [results[-1] * punishment_function(slope), point[1]]

