import multiprocessing as mp
import numpy as np

from evolving_classifier.FitnessCalculator import FitnessCalculator
from evolving_classifier.FitnessFunction import FitnessFunction
from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint


class MockFC(FitnessCalculator):
    def __init__(self):
        super().__init__()

    def compute(self, pool: mp.Pool, to_compute: [ChaosNet], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        pass