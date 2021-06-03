from statistics import mean

from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff, Softmax
from neural_network.FeedForwardNeuralNetwork import *
import numpy as np

from sklearn.linear_model import LinearRegression


class FitnessFunction:
    def __init__(self, learningIts):
        self.learningIts = learningIts
        pass

    def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, float]:
        pass

class ProgressFF(FitnessFunction):
    def __init__(self, learningIts):
        super().__init__(learningIts)

    def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, float]:
        network = network_from_point(point, seed)

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

        return [results[-1] * punishment_function(slope), results[-1]]

class ProgressFF2(FitnessFunction):
    def __init__(self, learningIts):
        super().__init__(learningIts)

    def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, float]:
        network = network_from_point(point, seed)

        network.train(inputs=trainInputs, outputs=trainOutputs, epochs=self.learningIts) #TODO could use validation probably
        test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
        main_eff = mean(test_results[0:3])

        partial_effs = []
        for i in range(len(network.cm_hist)):
            partial_effs.append(efficiency(network.cm_hist[i]))

        y = np.array(partial_effs)
        x = np.array(list(range(0, len(partial_effs))))

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        reg = LinearRegression().fit(x, y)
        slope = reg.coef_

        return [main_eff * punishment_function(slope), main_eff]

class PureEfficiencyFF(FitnessFunction):
    def __init__(self, learningIts):
        super().__init__(learningIts)

    def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, float]:
        network = network_from_point(point, seed)

        results = []

        for i in range(self.learningIts):
            network.train(inputs=trainInputs, outputs=trainOutputs, epochs=1) #TODO could use validation probably
            test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
            result = mean(test_results[0:3])
            results.append(result)

        return [results[-1], results[-1]]

class PureProgressFF(FitnessFunction):
    def __init__(self, learningIts):
        super().__init__(learningIts)

    def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, float]:
        network = network_from_point(point, seed)

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

        return [punishment_function(slope), results[-1]]

