import multiprocessing as mp
import random
from math import ceil

from evolving_classifier import FitnessFunction
import numpy as np

from neural_network.LsmNetwork import LsmNetwork
from utility.CNDataPoint import CNDataPoint


class FitnessCalculator:
    def __init__(self):
        pass

    def compute(self, pool: mp.Pool, to_compute: [LsmNetwork], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        pass

class LsmFitnessCalculator(FitnessCalculator):
    def __init__(self):
        super().__init__()

    def compute(self, pool: mp.Pool, to_compute: [LsmNetwork], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        results = [CNDataPoint(point) for point in to_compute]

        if pool is None:
            new_fitnesses = [fitnessFunc.compute(results[i].net, trainInputs, trainOutputs)for i in range(len(results))]
        else:
            estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(results[i].net, trainInputs, trainOutputs)) for i in range(len(results))]
            [estimation_result.wait() for estimation_result in estimating_async_results]
            new_fitnesses = [result.get() for result in estimating_async_results]

        for i in range(len(results)):
            results[i].add_data(new_fitnesses[i][0], new_fitnesses[i][1])

        for i in range(len(results)):
            if np.isnan(results[i].ff[0]):
                results[i].ff[0] = -np.inf

        for i in reversed(range(len(results[0].ff))):
            results = sorted(results, key=lambda x: x.ff[i], reverse=True)

        return results
