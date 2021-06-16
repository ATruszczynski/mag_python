import multiprocessing as mp
import random
from math import ceil

from ann_point import AnnPoint
from evolving_classifier import FitnessFunction
import numpy as np

from neural_network.ChaosNet import ChaosNet
from neural_network.FeedForwardNeuralNetwork import efficiency
from utility.AnnDataPoint import AnnDataPoint



class FitnessCalculator:
    def __init__(self):
        pass

    def compute(self, pool: mp.Pool, to_compute: [ChaosNet], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]):
        pass

class CNFitnessCalculator(FitnessCalculator):
    def __init__(self):
        super().__init__()

    def compute(self, pool: mp.Pool, to_compute: [ChaosNet], fitnessFunc: FitnessFunction,
                trainInputs: [np.ndarray], trainOutputs: [np.ndarray]):
        count = len(to_compute)

        estimates = [[point, AnnDataPoint(point)] for point in to_compute]#TODO moża stąd wyrzucić point

        estimates = sorted(estimates, key=lambda x: x[1].ff, reverse=True)
        seeds = [random.randint(0, 1000) for i in range(len(to_compute))]

        if pool is None:
            new_fitnesses = [fitnessFunc.compute(to_compute[i], trainInputs, trainOutputs, seeds[i])for i in range(len(to_compute))]
        else:
            estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(to_compute[i], trainInputs, trainOutputs, seeds[i])) for i in range(len(to_compute))]
            [estimation_result.wait() for estimation_result in estimating_async_results]
            new_fitnesses = [result.get() for result in estimating_async_results]

        for i in range(len(to_compute)):
            estimates[i][1].add_data(new_fitnesses[i][0], new_fitnesses[i][1])

        return estimates


#
# class OnlyFitnessCalculator(FitnessCalculator):
#     def __init__(self, fractions: [float]):
#         super().__init__()
#         self.fractions = fractions
#         pass
#
#     def compute(self, pool: mp.Pool, to_compute: [AnnPoint], fitnessFunc: FitnessFunction,
#                 trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [AnnPoint, AnnDataPoint]:
#         count = len(to_compute)
#
#         estimates = [[point, AnnDataPoint(point)] for point in to_compute]#TODO moża stąd wyrzucić point
#
#         for f in range(len(self.fractions)):
#             frac = self.fractions[f]
#
#             estimates = sorted(estimates, key=lambda x: x[1].ff, reverse=True)
#             comp_count = ceil(frac * count)
#             to_compute = [est[0] for est in estimates[0:comp_count]]
#             seeds = [random.randint(0, 1000) for i in range(len(to_compute))]
#
#             if pool is None:
#                 new_fitnesses = [fitnessFunc.compute(to_compute[i], trainInputs, trainOutputs, seeds[i])for i in range(len(to_compute))]
#             else:
#                 estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(to_compute[i], trainInputs, trainOutputs, seeds[i])) for i in range(len(to_compute))]
#                 [estimation_result.wait() for estimation_result in estimating_async_results]
#                 new_fitnesses = [result.get() for result in estimating_async_results]
#
#             for i in range(comp_count):
#                 estimates[i][1].add_data(new_fitnesses[i][0], new_fitnesses[i][1])
#
#         return estimates
#
# class PlusSizeFitnessCalculator(FitnessCalculator):
#     def __init__(self, fractions: [float], max_size_pun: float):
#         super().__init__()
#         self.fractions = fractions
#         self.max_size_pun = max_size_pun
#         pass
#
#     def compute(self, pool: mp.Pool, to_compute: [AnnPoint], fitnessFunc: FitnessFunction,
#                 trainInputs: [np.ndarray], trainOutputs: [np.ndarray]) -> [AnnPoint, AnnDataPoint]:
#         count = len(to_compute)
#
#         estimates = [[point, AnnDataPoint(point)] for point in to_compute]
#
#         for f in range(len(self.fractions)):
#             frac = self.fractions[f]
#
#             estimates = sorted(estimates, key=lambda x: x[1].ff, reverse=True)
#             comp_count = ceil(frac * count)
#             to_compute = [est[0] for est in estimates[0:comp_count]]
#             seeds = [random.randint(0, 1000) for i in range(len(to_compute))]
#
#             if pool is None:
#                 new_fitnesses = [fitnessFunc.compute(to_compute[i], trainInputs, trainOutputs, seeds[i])for i in range(len(to_compute))]
#             else:
#                 estimating_async_results = [pool.apply_async(func=fitnessFunc.compute, args=(to_compute[i], trainInputs, trainOutputs, seeds[i])) for i in range(len(to_compute))]
#                 [estimation_result.wait() for estimation_result in estimating_async_results]
#                 new_fitnesses = [result.get() for result in estimating_async_results]
#
#             for i in range(comp_count):
#                 estimates[i][1].add_data(new_fitnesses[i][0], new_fitnesses[i][1])
#
#         size_puns = np.linspace(1, self.max_size_pun, len(estimates))
#         eval_pop_sorted = sorted(estimates, key=lambda x: x[0].size())
#         for i in range(len(size_puns)):
#             eval_pop_sorted[i][1].ff *= size_puns[i]
#
#         return eval_pop_sorted