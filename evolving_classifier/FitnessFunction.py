from math import sqrt
from statistics import mean

# from ann_point.AnnPoint2 import CrossEntropy, AnnPoint2, QuadDiff, Softmax
from neural_network.ChaosNet import *
# from neural_network.FeedForwardNeuralNetwork import *
import numpy as np
from ann_point.Functions import *

from sklearn.linear_model import LinearRegression

class FitnessFunction:
    def __init__(self):
        pass

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        pass

class CNFF(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)
        eff = efficiency(test_results[0])

        return [[eff], test_results[0]]

class CNFF2(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[0])

        return [[(1 - eff) * -test_results[1]], test_results[0]]

class CNFF3(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[0])

        return [[(1 - eff)**2 * -test_results[1]], test_results[0]]

# TODO - C - could take type as argument
class CNFF4(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        return [[-test_results[1], -net.neuron_count], test_results[0]]

class CNFF5(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]
        eff = efficiency(cm)
        meff = m_efficiency(cm)

        result = mean([eff, meff])

        return [[result], cm]

class CNFF6(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        cm = test_results[0]
        eff = efficiency(cm)
        meff = m_efficiency(cm)

        mmeff = mean([eff, meff])

        lf = test_results[1]


        return [[-(1 - meff) * lf, -lf], cm]

class FFPUNEFF(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]

        pun = cm[0, 1] + 5 * cm[1, 0]

        return [[-pun, average_f1_score(cm), -net.neuron_count], cm]

class FFPUNQD(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        cm = test_results[0]

        result = (cm[0, 1] + 5 * cm[1, 0])
        result = -result * test_results[1]

        return [[result, -net.neuron_count], cm]

# class CNFF9(FitnessFunction):
#     def __init__(self):
#         super().__init__()
#
#     def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
#         test_results = net.test(test_input=trainInputs, test_output=trainOutputs)
#
#         cm = test_results[0]
#
#         result = cm[0, 1] + 5 * cm[1, 0]
#         result = -result
#
#         return [[result], cm]

# class CNF1(FitnessFunction):
#     def __init__(self):
#         super().__init__()
#
#     def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
#         test_results = net.test(test_input=trainInputs, test_output=trainOutputs)
#
#         cm = test_results[0]
#
#         result = average_f1_score(cm)
#
#         return [[result, -net.neuron_count], cm]

# class CNFFT(FitnessFunction):
#     def __init__(self, lossFun: LossFun):
#         super().__init__()
#         self.lossFun = lossFun
#
#     def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
#         test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
#
#         cm = test_results[0]
#         qd = test_results[1]
#
#         result = [average_f1_score(cm), -qd]
#
#         return [result, cm]

# class CNFFT2(FitnessFunction):
#     def __init__(self, lossFun: LossFun):
#         super().__init__()
#         self.lossFun = lossFun
#
#     def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
#         test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
#
#         cm = test_results[0]
#         qd = test_results[1]
#
#         result = [0, -qd]
#
#         return [result, cm]

class MEFF(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]

        return [[m_efficiency(cm), average_f1_score(cm), -net.neuron_count], cm]

class AVMAX(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        cm = test_results[0]
        av_qd = test_results[1]
        min_qd = test_results[2]

        qd = (av_qd + min_qd)/2

        return [[-qd, -net.neuron_count], cm]




