import math
from math import ceil
from statistics import mean, stdev
import warnings

from ann_point.AnnPoint import *
from ann_point.HyperparameterRange import *
import multiprocessing as mp

from evolving_classifier.EC_supervisor import EC_supervisor
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
from evolving_classifier.operators.HillClimbOperator import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from neural_network.FeedForwardNeuralNetwork import *
import random
from sklearn.linear_model import LinearRegression

from utility.Mut_Utility import *
from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

def_frac = [1, 0.6, 0.4, 0.25, 0.15, 0.1]

class EvolvingClassifier:
    def __init__(self, hrange: HyperparameterRange = None, logs_path: str = ""):
        if hrange is None:
            self.hrange = get_default_hrange()
        else:
            self.hrange = hrange

        # self.fractions = [1, 0.6, 0.4, 0.25, 0.15, 0.1]
        if logs_path == "":
            self.supervisor = EC_supervisor(logs)
        else:
            self.supervisor = EC_supervisor(logs_path)

        #TODO remove needless things from here
        self.av_dist = 0
        self.population = []
        self.pop_size = -1
        self.tournament_size = 2
        self.trainInputs = []
        self.trainOutputs = []
        self.testInputs = []
        self.testOutputs = []
        self.learningIts = 20

        self.mut_performed = 0
        self.cross_performed = 0

        self.co = SimpleCrossoverOperator(self.hrange)
        self.mo = SimpleAndStructuralCNMutation(self.hrange, 2)
        self.so = TournamentSelection(4)
        self.ff = CNFF()
        self.fc = CNFitnessCalculator()


    def prepare(self, popSize:int, startPopSize: int,
                nn_data: ([np.ndarray], [np.ndarray]), hidden_size: int, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        self.pop_size = popSize
        self.trainInputs = nn_data[0]
        self.trainOutputs = nn_data[1]
        input_size = self.trainInputs[0].shape[0]
        output_size = self.trainOutputs[0].shape[0]
        self.population = generate_population(self.hrange, startPopSize, input_size=input_size, output_size=output_size)

    def run(self, iterations: int, pm: float, pc: float, power: int = 1) -> ChaosNet:
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None
        self.supervisor.start(iterations)

        input_size = self.trainInputs[0].shape[0]
        output_size = self.trainOutputs[0].shape[0]

        self.supervisor.get_algo_data(input_size=input_size, output_size=output_size, pmS=pm, pmE=pm,
                                      pcS=pc, pcE=pc,
                                      ts=self.tournament_size, sps=len(self.population),
                                      ps=self.pop_size, fracs=[0], hrange=self.hrange, learningIts=self.ff.learningIts)

        self.supervisor.start(iterations=iterations)

        wb = np.linspace(0.1, 0.001, iterations)
        wbrr = 0.05
        # p = np.linspace(0.25, 0.01)
        best = [self.population[0], -math.inf]

        for i in range(iterations):
            if i == 20:
                ori = 1
            # eval_pop = self.calculate_fitnesses(pool, self.population)
            eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                       trainOutputs=self.trainOutputs)

            sorted_eval = sorted(eval_pop, key=lambda x: x[1].ff, reverse=True)

            if sorted_eval[0][1].ff >= best[1]:
                best = [sorted_eval[0][0].copy(), sorted_eval[0][1].ff]


            eval_pop = [eval_pop[i] for i in range(len(eval_pop)) if not np.isnan(eval_pop[i][1].ff)]
            # print(sorted_eval[0][0].weights)
            mean_eff = mean([x[1].get_eff() for x in eval_pop])
            mean_mut_rad = mean([x[0].mutation_radius for x in eval_pop])
            mean_wb_mut_prob = mean([x[0].wb_mutation_prob for x in eval_pop])
            mean_s_mut_prob = mean([x[0].s_mutation_prob for x in eval_pop])
            print(f"mean_mut_rad: {round(mean_mut_rad, 3)}|mean_wb_prob: {round(mean_wb_mut_prob, 3)}|mean_s_prob: {round(mean_s_mut_prob, 3)}")

            self.supervisor.check_point(eval_pop, i)
            crossed = []


            while len(crossed) < self.pop_size:
                c1 = self.so.select(val_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= pc:
                    c2 = self.so.select(val_pop=eval_pop)
                    cr_result = self.co.crossover(c1, c2)
                    crossed.extend(cr_result)
                    self.cross_performed += 1
                else:
                    crossed.append(c1)

            new_pop = []

            for ind in range(len(crossed)):
                # new_pop.append(self.mo.mutate(crossed[ind], pm=pm, radius=mut_rad))
                to_mutate = crossed[ind] #TODO to podawanie argumentów tutaj nie ma sensu
                new_pop.append(self.mo.mutate(to_mutate, wb_pm=to_mutate.wb_mutation_prob, s_pm=to_mutate.s_mutation_prob, p_pm=to_mutate.p_mutation_prob, radius=to_mutate.mutation_radius))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=to_mutate.edge_count() * to_mutate.wb_mutation_prob,
                #                                          s_pm=to_mutate.s_mutation_prob / self.pop_size,
                #                                          p_pm=to_mutate.p_mutation_prob / self.pop_size,
                #                                          radius=to_mutate.mutation_radius))




                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=wb[i], s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=wb[i] * (1 - mean_eff), s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=1/to_mutate.edge_count() * (1 - mean_eff), s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=3/to_mutate.edge_count() * (1 - mean_eff)**2, s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=0.01, s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=wbrr * (1 - mean_eff), s_pm=0, p_pm=0, radius=0))
                # new_pop.append(self.mo.mutate(to_mutate, wb_pm=0.1 * (1 - mean_eff), s_pm=0, p_pm=0, radius=0))


            self.population = new_pop
            print(f"best ff: {best[1]}")


            # if i % 75 == 0:
            #     for dd in range(len(self.population)):
            #         self.population[dd].mutation_radius = random.uniform(self.hrange.min_mut_radius, self.hrange.max_mut_radius)
            #         self.population[dd].wb_mutation_prob = random.uniform(self.hrange.min_wb_mut_prob, self.hrange.max_wb_mut_prob)
            #         self.population[dd].s_mutation_prob = random.uniform(self.hrange.min_s_mut_prob, self.hrange.max_s_mut_prob)
            #         self.population[dd].p_mutation_prob = random.uniform(self.hrange.min_p_mut_prob, self.hrange.max_p_mut_prob)

            # self.population = [sorted_eval[i][0] for i in range(5)]
            # self.population.extend(new_pop[:-5])

        # eval_pop = self.calculate_fitnesses(pool, self.population)
        eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                   trainOutputs=self.trainOutputs)
        self.supervisor.check_point(eval_pop, iterations)
        if power > 1:
            pool.close()

        sorted_eval = sorted(eval_pop, key=lambda x: x[1].ff, reverse=True)
        if sorted_eval[0][1].ff >= best[1]:
            best = [sorted_eval[0][0].copy(), sorted_eval[0][1].ff]

        return best[0]

    # def select(self, eval_pop: [[AnnPoint, float]]) -> AnnPoint2:
    #     chosen = choose_without_repetition(options=eval_pop, count=self.tournament_size)
    #     chosen_sorted = sorted(chosen, key=lambda x: x[1], reverse=True)
    #     return chosen_sorted[0][0]

    # def crossover(self, pointA: AnnPoint2, pointB: AnnPoint2, cross_probs: [float]) -> [AnnPoint2]:
    #     pointA = pointA.copy()
    #     pointB = pointB.copy()
    #
    #     if len(pointA.hidden_neuron_counts) < len(pointB.hidden_neuron_counts):
    #         tmp = pointA
    #         pointA = pointB
    #         pointB = tmp
    #
    #     layersA = pointA.into_numbered_layer_tuples()
    #     layersB = pointB.into_numbered_layer_tuples()
    #
    #     choices = []
    #     for i in range(1, len(layersA) - 1):
    #         for j in range(1, len(layersB) - 1):
    #             choices.append((i, j))
    #     choices.append((len(layersA) - 1, len(layersB) - 1))
    #
    #     choice = choices[random.randint(0, len(choices) - 1)]
    #
    #
    #     layAInd = choice[0]
    #     layBInd = choice[1]
    #
    #     tmp = layersA[layAInd]
    #     layersA[layAInd] = layersB[layBInd]
    #     layersB[layBInd] = tmp
    #
    #     for i in range(1, len(layersA)):
    #         layer = layersA[i]
    #         pre_layer = layersA[i - 1]
    #         if pre_layer[1] != layer[3].shape[1]:
    #             layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
    #             layer[4] = np.zeros((layer[1], 1))
    #         layersA[i] = layer
    #
    #     for i in range(1, len(layersB)):
    #         layer = layersB[i]
    #         pre_layer = layersB[i - 1]
    #         if pre_layer[1] != layer[3].shape[1]:
    #             layer[3] = get_Xu_matrix((layer[1], pre_layer[1]))
    #             layer[4] = np.zeros((layer[1], 1))
    #         layersB[i] = layer
    #
    #     pointA = point_from_layer_tuples(layersA)
    #     pointB = point_from_layer_tuples(layersB)
    #
    #     return [pointA, pointB]
    #
    #     # if cross_probs[0] < 0.5:
    #     #     tmp = pointA.hiddenLayerCount
    #     #     pointA.hiddenLayerCount = pointB.hiddenLayerCount
    #     #     pointB.hiddenLayerCount = tmp
    #     #
    #     # if cross_probs[1] < 0.5:
    #     #     tmp = pointA.neuronCount
    #     #     pointA.neuronCount = pointB.neuronCount
    #     #     pointB.neuronCount = tmp
    #     #
    #     # if cross_probs[2] < 0.5:
    #     #     tmp = pointA.actFun.copy()
    #     #     pointA.actFun = pointB.actFun.copy()
    #     #     pointB.actFun = tmp
    #     #
    #     # if cross_probs[3] < 0.5:
    #     #     tmp = pointA.aggrFun.copy()
    #     #     pointA.aggrFun = pointB.aggrFun.copy()
    #     #     pointB.aggrFun = tmp
    #     #
    #     # if cross_probs[4] < 0.5:
    #     #     tmp = pointA.lossFun.copy()
    #     #     pointA.lossFun = pointB.lossFun.copy()
    #     #     pointB.lossFun = tmp
    #     #
    #     # if cross_probs[5] < 0.5:
    #     #     tmp = pointA.learningRate
    #     #     pointA.learningRate = pointB.learningRate
    #     #     pointB.learningRate = tmp
    #     #
    #     # if cross_probs[6] < 0.5:
    #     #     tmp = pointA.momCoeff
    #     #     pointA.momCoeff = pointB.momCoeff
    #     #     pointB.momCoeff = tmp
    #     #
    #     # if cross_probs[7] < 0.5:
    #     #     tmp = pointA.batchSize
    #     #     pointA.batchSize = pointB.batchSize
    #     #     pointB.batchSize = tmp
    #
    #     return [pointA, pointB]
    #
    # def mutate(self, point: AnnPoint2, pm_wb: float, pm_sc: float, radius: float) -> AnnPoint2:
    #     point = point.copy()
    #
    #     # Zmień liczbę layerów
    #     structural = False
    #
    #     if random.random() < pm_sc:
    #         current = len(point.hidden_neuron_counts)
    #         minhl = max(current - 1, self.hrange.layerCountMin)
    #         maxhl = min(current + 1, self.hrange.layerCountMax)
    #         new = try_choose_different(current, list(range(minhl, maxhl + 1)))
    #
    #         point = change_amount_of_layers(point=point, demanded=new, hrange=self.hrange)
    #         structural = True
    #
    #     # Zmień county neuronów
    #     for i in range(len(point.hidden_neuron_counts)):
    #         if random.random() < pm_sc:
    #             current = point.hidden_neuron_counts[i]
    #             new = try_choose_different(current, list(range(self.hrange.neuronCountMin, self.hrange.neuronCountMax + 1)))
    #             point = change_neuron_count_in_layer(point=point, layer=i, demanded=new)
    #             structural = True
    #
    #     # Zmień funkcje
    #     for i in range(len(point.hidden_neuron_counts)):
    #         if random.random() < pm_sc:
    #             current = point.activation_functions[i]
    #             new = try_choose_different(current, self.hrange.actFunSet)
    #             point.activation_functions[i] = new.copy()
    #             structural = True
    #
    #     # Zmień wagi
    #
    #     for i in range(len(point.weights)):
    #         if random.random() < pm_wb:
    #             for r in range(point.weights[i].shape[0]):
    #                 for c in range(point.weights[i].shape[1]):
    #                     point.weights[i][r, c] += random.gauss(0, radius / sqrt(point.weights[i].shape[1]))
    #     # Zmień biasy
    #     for i in range(len(point.biases)):
    #         if random.random() < pm_wb:
    #             for r in range(point.biases[i].shape[0]):
    #                 point.biases[i][r] += random.gauss(0, radius)
    #
    #     # if structural:
    #         # network = network_from_point(point, seed=random.randint(0, 1000))
    #         # network.train(self.trainInputs, self.trainOutputs, 1)
    #         # for i in range(1, len(network.weights)):
    #         #     point.weights[i - 1] = network.weights[i].copy()
    #         #     point.biases[i - 1] = network.biases[i].copy()
    #         # print("struc")
    #
    #     return point
    #
    #     if mut_probs[0] < pm_wb * radius:
    #         current = point.hiddenLayerCount
    #         minhl = max(current - 1, self.hrange.layerCountMin)
    #         maxhl = min(current + 1, self.hrange.layerCountMax)
    #         point.hiddenLayerCount = try_choose_different(point.hiddenLayerCount, list(range(minhl, maxhl + 1)))
    #         self.mut_performed += 1
    #
    #     if mut_probs[1] < pm_wb:
    #         point.neuronCount = get_in_radius(point.neuronCount, self.hrange.neuronCountMin, self.hrange.neuronCountMax, radius)
    #         self.mut_performed += 1
    #
    #     if mut_probs[2] < pm_wb * radius:
    #         point.actFun = try_choose_different(point.actFun, self.hrange.actFunSet)
    #         self.mut_performed += 1
    #
    #     if mut_probs[3] < pm_wb * radius:
    #         point.aggrFun = try_choose_different(point.aggrFun, self.hrange.aggrFunSet)
    #         self.mut_performed += 1
    #
    #     if mut_probs[4] < pm_wb * radius:
    #         point.lossFun = try_choose_different(point.lossFun, self.hrange.lossFunSet)
    #         self.mut_performed += 1
    #
    #     if mut_probs[5] < pm_wb:
    #         point.learningRate = get_in_radius(point.learningRate, self.hrange.learningRateMin, self.hrange.learningRateMax, radius)
    #         self.mut_performed += 1
    #
    #     if mut_probs[6] < pm_wb:
    #         point.momCoeff = get_in_radius(point.momCoeff, self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax, radius)
    #         self.mut_performed += 1
    #
    #     if mut_probs[7] < pm_wb:
    #         point.batchSize = get_in_radius(point.batchSize, self.hrange.batchSizeMin, self.hrange.batchSizeMax, radius)
    #         self.mut_performed += 1
    #
    #     return point

    # def calculate_fitnesses(self, pool: mp.Pool, to_compute: [AnnPoint]) -> [[AnnPoint, float]]:
    #     count = len(to_compute)
    #
    #     touches = [0] * count
    #
    #     estimates = [[point, 0] for point in to_compute]
    #
    #     for f in range(len(self.fractions)):
    #         frac = self.fractions[f]
    #
    #         estimates = sorted(estimates, key=lambda x: x[1], reverse=True)
    #         comp_count = ceil(frac * count)
    #         to_compute = [est[0] for est in estimates[0:comp_count]]
    #         seeds = [random.randint(0, 1000) for i in range(len(to_compute))]
    #
    #         if pool is None:
    #             new_fitnesses = [self.ff.compute(to_compute[i], self.trainInputs, self.trainOutputs, seeds[i])for i in range(len(to_compute))]
    #         else:
    #             estimating_async_results = [pool.apply_async(func=self.ff.compute, args=(to_compute[i], self.trainInputs, self.trainOutputs, seeds[i])) for i in range(len(to_compute))]
    #             [estimation_result.wait() for estimation_result in estimating_async_results]
    #             new_fitnesses = [result.get() for result in estimating_async_results]
    #
    #         for i in range(comp_count):
    #             new_fit = new_fitnesses[i]
    #             curr_est = estimates[i][1]
    #             touch = touches[i]
    #
    #             curr_sum = curr_est * touch
    #             new_sum = curr_sum + new_fit
    #             new_est = new_sum / (touch + 1)
    #
    #             estimates[i][1] = new_est
    #             touches[i] += 1
    #
    #     return estimates

        # estimates = sorted(estimates, key=lambda x: x[0].size(), reverse=True)
        # size_puns = np.linspace(0.95, 1, len(estimates))
        # for i in range(len(estimates)):
        #     estimates[i][1] *= size_puns[i]

        # dist = average_distance_between_points([e[0] for e in estimates], self.hrange)
        # self.av_dist = mean(dist)
        # print(stdev(dist))
        #
        #
        # points = [[to_compute[i], i] for i in range(len(to_compute))]
        #
        # if pool is None:
        #     new_fitnesses = [self.ff.compute(points[i], self.trainInputs, self.trainOutputs)for i in range(len(points))]
        # else:
        #     estimating_async_results = [pool.apply_async(func=self.ff.compute, args=(points[i], self.trainInputs, self.trainOutputs)) for i in range(len(points))]
        #     [estimation_result.wait() for estimation_result in estimating_async_results]
        #     new_fitnesses = [result.get() for result in estimating_async_results]
        #
        #
        # estimates = [[to_compute[i[1]], i[0]] for i in new_fitnesses]
        #
        # return estimates

    # def calculate_fitness(self, point: AnnPoint2):
    #     network = network_from_point(point, 1001)
    #
    #
    #     test_results = network.test(test_input=self.trainInputs, test_output=self.trainOutputs) #TODO DONT USE TEST SETS IN TRAINING PROCESS WTF
    #     result = mean(test_results[0:3])
    #
    #
    #     # y = np.array(results)
    #     # x = np.array(list(range(0, self.learningIts)))
    #     #
    #     # x = x.reshape((-1, 1))
    #     # y = y.reshape((-1, 1))
    #     #
    #     # reg = LinearRegression().fit(x, y)
    #     # slope = reg.coef_
    #     # ff = results[-1] #* punishment_function(slope)
    #     # print(f"{point.to_string()} - {point.size()} - {result}")
    #     return result



