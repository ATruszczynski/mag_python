import multiprocessing as mp
import random
from math import ceil

from evolving_classifier import FitnessFunction
import numpy as np

from neural_network.ChaosNet import ChaosNet
from utility.CNDataPoint import CNDataPoint


class FitnessCalculator:
    def __init__(self):
        pass

    def compute(self, pool: mp.Pool, to_compute: [ChaosNet], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        pass

class CNFitnessCalculator(FitnessCalculator):
    def __init__(self):
        super().__init__()

    def compute(self, pool: mp.Pool, to_compute: [ChaosNet], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [CNDataPoint]:
        results = [CNDataPoint(point) for point in to_compute]

        if pool is None:
            new_fitnesses = [fitnessFunc.compute(results[i].net.copy(), trainInputs, trainOutputs)for i in range(len(results))]
        else:
            estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(results[i].net, trainInputs, trainOutputs)) for i in range(len(results))]
            [estimation_result.wait() for estimation_result in estimating_async_results]
            new_fitnesses = [result.get() for result in estimating_async_results]

        for i in range(len(to_compute)):
            results[i].add_data(new_fitnesses[i][0], new_fitnesses[i][1])

        for i in range(len(results)):
            if np.isnan(results[i].ff):
                results[i].ff = -np.inf

        results = sorted(results, key=lambda x: x.ff, reverse=True)

        return results
