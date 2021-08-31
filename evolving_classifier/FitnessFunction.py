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

        return [eff, test_results[0]]

class CNFF2(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[0])

        return [(1 - eff) * -test_results[1], test_results[0]]

class CNFF3(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)
        eff = efficiency(test_results[0])

        return [(1 - eff)**2 * -test_results[1], test_results[0]]

# TODO - C - could take type as argument
class CNFF4(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        return [-test_results[1], test_results[0]]

class CNFF5(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]
        eff = efficiency(cm)
        meff = m_efficiency(cm)

        result = (9. * meff + eff) / 10.

        return [result, cm]

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

        mmeff = (9. * meff + eff) / 10.

        result = -(1.00 - mmeff + 1e-3)**4 * test_results[1]

        return [result, cm]

# TODO - A - test
class CNFF7(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]

        result = cm[0, 1] + 5 * cm[1, 0]
        result = (1.01 - m_efficiency(cm)) * result
        result = -result

        return [result, cm]

# TODO - A - test
class CNFF8(FitnessFunction):
    def __init__(self, lossFun: LossFun):
        super().__init__()
        self.lossFun = lossFun

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs, lf=self.lossFun)

        cm = test_results[0]

        result = (cm[0, 1] + 5 * cm[1, 0] + 1)
        result = -result * test_results[1]

        return [result, cm]


# TODO - A - test
class CNFF9(FitnessFunction):
    def __init__(self):
        super().__init__()

    def compute(self, net: ChaosNet, trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [float, np.ndarray]:
        test_results = net.test(test_input=trainInputs, test_output=trainOutputs)

        cm = test_results[0]

        result = cm[0, 1] + 5 * cm[1, 0]
        result = -result

        return [result, cm]



