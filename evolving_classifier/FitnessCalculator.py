import multiprocessing as mp
import random
from math import ceil

from ann_point import AnnPoint
from evolving_classifier import FitnessFunction
import numpy as np

from neural_network.FeedForwardNeuralNetwork import efficiency
from utility.AnnDataPoint import AnnDataPoint


class FitnessCalculator:
    def __init__(self):
        pass

    def compute(self, pool: mp.Pool, to_compute: [AnnPoint], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]):
        pass

class OnlyFitnessCalculator(FitnessCalculator):
    def __init__(self, fractions: [float]):
        super().__init__()
        self.fractions = fractions
        pass

    def compute(self, pool: mp.Pool, to_compute: [AnnPoint], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [AnnPoint, AnnDataPoint]:
        count = len(to_compute)

        estimates = [[point, AnnDataPoint(point)] for point in to_compute]

        for f in range(len(self.fractions)):
            frac = self.fractions[f]

            estimates = sorted(estimates, key=lambda x: x[1].ff, reverse=True)
            comp_count = ceil(frac * count)
            to_compute = [est[0] for est in estimates[0:comp_count]]
            seeds = [random.randint(0, 1000) for i in range(len(to_compute))]

            if pool is None:
                new_fitnesses = [fitnessFunc.compute(to_compute[i], trainInputs, trainOutputs, seeds[i])for i in range(len(to_compute))]
            else:
                estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(to_compute[i], trainInputs, trainOutputs, seeds[i])) for i in range(len(to_compute))]
                [estimation_result.wait() for estimation_result in estimating_async_results]
                new_fitnesses = [result.get() for result in estimating_async_results]

            for i in range(comp_count):
                # new_fit = new_fitnesses[i][0]
                # new_eff = efficiency(new_fitnesses[i][1])
                # curr_est = estimates[i][1]
                # curr_est_eff = estimates[i][2]
                # touch = touches[i]
                #
                # curr_sum = curr_est * touch
                # new_sum = curr_sum + new_fit
                # new_est = new_sum / (touch + 1)
                #
                # curr_sum_eff = curr_est_eff * touch
                # new_sum_eff = curr_sum_eff + new_eff
                # new_est_eff = new_sum_eff / (touch + 1)
                #
                # estimates[i][1] = new_est
                # estimates[i][2] = new_est_eff
                # touches[i] += 1
                estimates[i][1].add_data(new_fitnesses[i][0], new_fitnesses[i][1])

        return estimates