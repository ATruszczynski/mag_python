from statistics import mean

from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff, Softmax
from neural_network.ChaosNet import ChaosNet
from neural_network.FeedForwardNeuralNetwork import *
import numpy as np

from sklearn.linear_model import LinearRegression


class FitnessFunction:
    def __init__(self, learningIts):
        self.learningIts = learningIts
        pass

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
        pass

class CNFF(FitnessFunction):
    def __init__(self):
        super().__init__(0)

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)
        eff = efficiency(test_results[3])

        return [eff, test_results[3]]

class CNFF3(FitnessFunction):
    def __init__(self):
        super().__init__(0)

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)
        eff = efficiency(test_results[3])

        return [eff**6 * sqrt(sqrt(net.s_mutation_prob)) * sqrt(sqrt(net.wb_mutation_prob)), test_results[3]]

class CNFF2(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__(0)
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[3])

        return [(1 - eff)**2 * -test_results[4], test_results[3]]
        # return [-test_results[4], test_results[3]]
        # return [eff, test_results[3]]

class CNFF4(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__(0)
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[3])

        return [-test_results[4], test_results[3]]
        # return [-test_results[4], test_results[3]]
        # return [eff, test_results[3]]

# class ProgressFF(FitnessFunction):
#     def __init__(self, learningIts):
#         super().__init__(learningIts)
#
#     def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
#         network = network_from_point(point, seed)
#
#         results = []
#         test_results = []
#
#         for i in range(self.learningIts):
#             network.train(inputs=trainInputs, outputs=trainOutputs, epochs=1)
#             test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
#             result = mean(test_results[0:3])
#             results.append(result)
#
#         y = np.array(results)
#         x = np.array(list(range(0, self.learningIts)))
#
#         x = x.reshape((-1, 1))
#         y = y.reshape((-1, 1))
#
#         reg = LinearRegression().fit(x, y)
#         slope = reg.coef_
#
#         return [results[-1] * punishment_function(slope), test_results[3]]
#
# class ProgressFF2(FitnessFunction):
#     def __init__(self, learningIts):
#         super().__init__(learningIts)
#
#     def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
#         network = network_from_point(point, seed)
#
#         network.train(inputs=trainInputs, outputs=trainOutputs, epochs=self.learningIts)
#         test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
#         main_eff = mean(test_results[0:3])
#
#         partial_effs = []
#         for i in range(len(network.cm_hist)):
#             partial_effs.append(efficiency(network.cm_hist[i]))
#
#         y = np.array(partial_effs)
#         x = np.array(list(range(0, len(partial_effs))))
#
#         x = x.reshape((-1, 1))
#         y = y.reshape((-1, 1))
#
#         reg = LinearRegression().fit(x, y)
#         slope = reg.coef_
#
#         return [main_eff * punishment_function(slope), test_results[3]]
#
# class PureEfficiencyFF(FitnessFunction):
#     def __init__(self, learningIts):
#         super().__init__(learningIts)
#
#     def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
#         network = network_from_point(point, seed)
#
#         results = []
#         test_results = []
#
#         for i in range(self.learningIts):
#             network.train(inputs=trainInputs, outputs=trainOutputs, epochs=1)
#             test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
#             result = mean(test_results[0:3])
#             results.append(result)
#
#         return [results[-1], test_results[3]]
#
# class PureProgressFF(FitnessFunction):
#     def __init__(self, learningIts):
#         super().__init__(learningIts)
#
#     def compute(self, point: AnnPoint, trainInputs: [np.ndarray], trainOutputs: [np.ndarray], seed: int) -> [float, np.ndarray]:
#         network = network_from_point(point, seed)
#
#         results = []
#         test_results = []
#
#         for i in range(self.learningIts):
#             network.train(inputs=trainInputs, outputs=trainOutputs, epochs=1)
#             test_results = network.test(test_input=trainInputs, test_output=trainOutputs)
#             result = mean(test_results[0:3])
#             results.append(result)
#
#         y = np.array(results)
#         x = np.array(list(range(0, self.learningIts)))
#
#         x = x.reshape((-1, 1))
#         y = y.reshape((-1, 1))
#
#         reg = LinearRegression().fit(x, y)
#         slope = reg.coef_
#
#         return [punishment_function(slope), test_results[3]]
#
