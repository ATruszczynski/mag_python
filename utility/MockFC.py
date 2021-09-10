import multiprocessing as mp
import numpy as np

from evolving_classifier.LsmFitnessCalculator import FitnessCalculator
from evolving_classifier.FitnessFunction import FitnessFunction
from neural_network.LsmNetwork import LsmNetwork
from utility.CNDataPoint import CNDataPoint


class MockFC(FitnessCalculator):
    def __init__(self):
        super().__init__()

    def compute(self, pool: mp.Pool, to_compute: [LsmNetwork], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        pass