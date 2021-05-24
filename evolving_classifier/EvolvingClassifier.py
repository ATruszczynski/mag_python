from math import ceil
from statistics import mean, stdev
import warnings

from ann_point.AnnPoint import *
from ann_point.HyperparameterRange import *
import multiprocessing as mp
import time

from evolving_classifier.EC_supervisor import EC_supervisor
from neural_network.FeedForwardNeuralNetwork import *
import random
from sklearn.linear_model import LinearRegression

from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

class EvolvingClassifier:
    def __init__(self, hrange: HyperparameterRange = None):
        if hrange is None:
            self.hrange = get_default_hrange()
        else:
            self.hrange = hrange

        self.population = []
        self.fractions = [1, 0.6, 0.4, 0.25, 0.15, 0.1]
        self.fractions = [1]
        self.pop_size = -1
        self.pc = -1
        self.pm = -1
        self.beg_pm = -1
        self.tournament_size = 2
        self.trainInputs = []
        self.trainOutputs = []
        self.testInputs = []
        self.testOutputs = []
        self.learningIts = 10
        self.batchSize = 25
        self.supervisor = EC_supervisor(logs)
        self.av_dist = 0
        self.mut_steps = []
        self.enough = 3 # TODO wyrzuć

        self.mut_performed = 0
        self.cross_performed = 0

    def prepare(self, popSize:int, startPopSize: int, pc: float, pm: float, tournament_size: int,
                nn_data: ([np.ndarray], [np.ndarray], [np.ndarray], [np.ndarray]), seed: int):
        random.seed(seed)
        self.pop_size = popSize
        self.pc = pc
        self.pm = pm
        self.beg_pm = pm
        self.tournament_size = tournament_size
        self.trainInputs = nn_data[0]
        self.trainOutputs = nn_data[1]
        self.testInputs = nn_data[2]
        self.testOutputs = nn_data[3]
        input_size = self.testInputs[0].shape[0]
        output_size = self.testOutputs[0].shape[0]
        self.population = generate_population(self.hrange, startPopSize, input_size=input_size, output_size=output_size)

        self.supervisor.get_algo_data(input_size=input_size, output_size=output_size, pm=self.pm, pc=self.pc,
                                      ts=self.tournament_size, sps=startPopSize,
                                      ps=self.pop_size, fracs=self.fractions, hrange=self.hrange)

    def run(self, iterations: int, power: int = 1) -> LooseNetwork:
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None

        tt = 0
        mt = 0
        ct = 0

        self.supervisor.start(iterations)
        self.mut_steps = np.logspace(1, 0, iterations, base=2)
        mut_radius = np.linspace(0.5, 0.5, iterations)
        pms = np.linspace(0.002, 0.002, iterations)
        # pms = np.linspace(0.000, 0.000, iterations)
        pcs = np.linspace(0.8, 0.8, iterations)

        print("start")
        # TODO cross entropy
        for i in range(iterations):
            s = time.time()
            eval_pop = self.calculate_fitnesses(pool, self.population)
            t = time.time()
            tt += t - s

            curr_max = max([e[1] for e in eval_pop])
            if curr_max >= self.enough:
                break

            # if self.av_dist < 1:
            #     self.pm = self.pm * self.mut_steps[i]
            # else:
            #     self.pm = self.beg_pm#pm / self.mut_steps[i]
            print(f"Av. dist.: {self.av_dist} - pm: {pms[i]} - {mut_radius[i]} - pc: {pcs[i]}")
            print(f"Mut: {self.mut_performed}, cross: {self.cross_performed}")

            self.supervisor.check_point(eval_pop, i)

            crossed = []

            s = time.time()
            while len(crossed) < self.pop_size:
                c1 = self.select(eval_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= pcs[i]:
                    c2 = self.select(eval_pop=eval_pop)
                    cr_result = self.crossover(c1, c2, [random.random() for i in range(8)])
                    crossed.extend(cr_result)
                    self.cross_performed += 1
                else:
                    crossed.append(c1)
            t = time.time()
            ct += t - s

            new_pop = []

            s = time.time()
            for ind in range(len(crossed)):
                new_pop.append(self.mutate(crossed[ind], pm=pms[i], radius=mut_radius[i], mut_probs=[random.random() for i in range(8)]))
            t = time.time()
            mt += t - s

            self.population = new_pop

        eval_pop = self.calculate_fitnesses(pool, self.population)
        if pool is not None:
            pool.close()

        eval_pop_sorted = sorted(eval_pop, key=lambda x: x[1], reverse=True)
        print(f"tt: {round(tt/iterations, 4)}, mt: {round(mt/iterations, 4)}, ct: {round(ct/iterations, 4)}")

        return eval_pop_sorted[0][0]

    def select(self, eval_pop: [[AnnPoint, float]]) -> AnnPoint:
        chosen = choose_without_repetition(options=eval_pop, count=self.tournament_size)
        chosen_sorted = sorted(chosen, key=lambda x: x[1], reverse=True)
        return chosen_sorted[0][0]

    def crossover(self, pointA: LooseNetwork, pointB: LooseNetwork, cross_probs: [float]) -> [LooseNetwork]:
        pointA = pointA.copy()
        pointB = pointB.copy()

        div = random.randint(pointA.hidden_start_index, pointA.neuron_count - 1)

        tmp = pointA.links[:, div]
        pointA.links[:, div] = pointB.links[:, div]
        pointB.links[:, div] = tmp

        tmp = pointA.weights[:, div]
        pointA.weights[:, div] = pointB.weights[:, div]
        pointB.weights[:, div] = tmp

        tmp = pointA.actFuns[div]
        pointA.actFuns[div] = pointB.actFuns[div]
        pointB.actFuns[div] = tmp

        tmp = pointA.bias[div]
        pointA.bias[div] = pointB.bias[div]
        pointB.bias[div] = tmp

        tmp = pointA.actFuns[div]
        pointA.actFuns[div] = pointB.actFuns[div]
        pointB.actFuns[div] = tmp

        # tmp = pointA.links[:, :div]
        # pointA.links[:, :div] = pointB.links[:, :div]
        # pointB.links[:, :div] = tmp
        #
        # tmp = pointA.weights[:, :div]
        # pointA.weights[:, :div] = pointB.weights[:, :div]
        # pointB.weights[:, :div] = tmp
        #
        # tmp = pointA.actFuns[:div]
        # pointA.actFuns[:div] = pointB.actFuns[:div]
        # pointB.actFuns[:div] = tmp
        #
        # tmp = pointA.bias[:div]
        # pointA.bias[:div] = pointB.bias[:div]
        # pointB.bias[:div] = tmp
        #
        # tmp = pointA.actFuns[:div]
        # pointA.actFuns[:div] = pointB.actFuns[:div]
        # pointB.actFuns[:div] = tmp

        return [pointA, pointB]

    def mutate(self, point: LooseNetwork, pm: float, radius: float, mut_probs: [float]) -> LooseNetwork:
        point = point.copy()

        # mutate each link
        for i in list(range(point.input_size, len(point.bias))):
            for j in range(0, min(i, len(point.bias) - point.output_size + 1)):
                mr = random.random()
                if mr < pm:
                    self.mut_performed += 1
                    link = point.links[j, i]
                    if link == 0:
                        point.links[j, i] = 1
                        point.weights[j, i] = random.gauss(mu=0, sigma=1)
                    else:
                        point.links[j, i] = 0
                        point.weights[j, i] = 0

        for i in list(range(point.input_size, len(point.bias))):
            for j in range(0, min(i, len(point.bias) - point.output_size + 1)):
                link = point.links[j, i]
                if link == 1:
                    mr = random.random()
                    if mr < pm:
                        self.mut_performed += 1
                        point.weights[j, i] += random.gauss(mu=0, sigma=radius)

        for i in point.get_col_indices():
            mr = random.random()
            if mr < pm:
                point.bias[i] += random.gauss(mu=0, sigma=radius)
            mr = random.random()
            if mr < pm:
                point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)

        # mutate each active weight
        # mutate each act func
        # mutate each bias

        # if mr < pm:
        #     self.mut_performed += 1
        #     choices = list(range(point.input_size, len(point.bias)))
        #     tot = len(choices)
        #     mut_count = ceil(tot * 0.1)
        #     to_mutate = choose_without_repetition(choices, mut_count)
        #
        #     for i in range(len(to_mutate)):
        #         ind = to_mutate[i]
        #         active = np.where(point.links[:, ind] == 0)[0]
        #
        #         mode = random.random()
        #         if mode < 0.1 * radius:
        #             to_deactivate_prob = np.random.uniform(size=(active.shape))
        #             to_deactivate = np.where(to_deactivate_prob < 0.1)[0]
        #             point.links[active[to_deactivate]] = 0
        #             point.weights[active[to_deactivate]] = 0
        #         elif mode < 0.2 * radius:
        #             weights = point.weights[active, ind]
        #             inactive = np.where(point.links[:, ind] == 1)[0]
        #             to_activate_prob = np.random.uniform(size=(inactive.shape))
        #             to_activate = np.where(to_activate_prob < 0.1)[0]
        #             avg = np.average(weights)
        #             sd = np.std(weights)
        #             point.links[inactive[to_activate], ind] = 1
        #             point.weights[inactive[to_activate], ind] += np.random.normal(avg, sd, size=(to_activate.shape))
        #         elif mode < 0.3 * radius:
        #             point.actFuns[ind] = try_choose_different(point.actFuns[ind], self.hrange.actFunSet)
        #         elif mode < 0.4:
        #             point.bias[ind] += random.gauss(mu=0, sigma=radius*0.1)
        #         else:
        #             weights = point.weights[active, ind]
        #             sd = np.std(weights)
        #             point.weights[active, ind] = np.random.normal(0, sd * radius, size=(active.shape))




        # for i in range(point.input_size, len(point.bias)):
        #     for j in range(0, i):
        #         mr = random.random()
        #         if mr < pm:
        #             link = point.links[j, i]
        #             if link == 0:
        #                 point.links[j, i] = 1
        #                 point.weights[j, i] = random.uniform(-0.1, 0.1) # TODO coś lepszego
        #             else:
        #                 mr = random.random()
        #                 if mr < 0.5:
        #                     point.weights[j, i] += random.gauss(0, radius * 0.1)
        #                 else:
        #                     point.links[j, i] = 0
        #                     point.weights[j, i] = 0
        #
        #             self.mut_performed += 1
        #
        #     mr = random.random()
        #     if mr < pm:
        #         point.actFuns[i] = try_choose_different(point.actFuns[i], self.hrange.actFunSet)
        #         self.mut_performed += 1
        #
        #     mr = random.random()
        #     if mr < pm:
        #         point.bias[i] += random.gauss(0, radius * 0.1)
        #         self.mut_performed += 1

        return point

    def calculate_fitnesses(self, pool: mp.Pool, points: [LooseNetwork]) -> [[LooseNetwork, float]]:
        # for l in points:
        #     l.analyse()
        count = len(points)

        touches = [0] * count

        estimates = [[point, 0] for point in points]

        for f in range(len(self.fractions)):
            frac = self.fractions[f]

            estimates = sorted(estimates, key=lambda x: x[1], reverse=True)
            comp_count = ceil(frac * count)
            to_compute = [est[0] for est in estimates[0:comp_count]]
            seeds = [random.randint(0, 1000) for i in range(len(to_compute))]

            if pool is None:
                new_fitnesses = [self.calculate_fitness(to_compute[i], seeds[i])for i in range(len(to_compute))]
            else:
                estimating_async_results = [pool.apply_async(func=self.calculate_fitness, args=(to_compute[i], seeds[i])) for i in range(len(to_compute))]
                [estimation_result.wait() for estimation_result in estimating_async_results]
                new_fitnesses = [result.get() for result in estimating_async_results]

            for i in range(comp_count):
                new_fit = new_fitnesses[i]
                curr_est = estimates[i][1]
                touch = touches[i]

                curr_sum = curr_est * touch
                new_sum = curr_sum + new_fit
                new_est = new_sum / (touch + 1)

                estimates[i][1] = new_est
                touches[i] += 1

        # estimates = sorted(estimates, key=lambda x: x[0].size(), reverse=True)
        #
        # size_puns = np.linspace(0.95, 1, len(estimates))
        # for i in range(len(estimates)):
        #     estimates[i][1] *= size_puns[i]

        # dist = average_distance_between_points([e[0] for e in estimates], self.hrange)
        # self.av_dist = mean(dist)
        # print(stdev(dist))

        return estimates

    def calculate_fitness(self, network: LooseNetwork, seed: int):
        test_results = network.test(test_input=self.testInputs, test_output=self.testOutputs)
        result = mean(test_results[0:3])

        return result



