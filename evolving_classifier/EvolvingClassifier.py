from math import ceil
from statistics import mean

from ann_point.AnnPoint import *
from ann_point.HyperparameterRange import *
import multiprocessing as mp

from evolving_classifier.EC_supervisor import EC_supervisor
from neural_network.FeedForwardNeuralNetwork import *
import random
from sklearn.linear_model import LinearRegression

from utility.Utility import *

logs = "logs"

class EvolvingClassifier:
    def __init__(self, hrange: HyperparameterRange = None):
        if hrange is None:
            self.hrange = get_default_hrange()
        else:
            self.hrange = hrange

        self.population = []
        self.fractions = [1, 0.75, 0.5, 0.25]
        self.pop_size = -1
        self.pc = -1
        self.pm = -1
        self.tournament_size = 2
        self.trainInputs = []
        self.trainOutputs = []
        self.testInputs = []
        self.testOutputs = []
        self.learningIts = 5
        self.batchSize = 25
        self.supervisor = EC_supervisor(logs)

    def prepare(self, popSize:int, startPopSize: int, pc: float, pm: float, tournament_size: int,
                nn_data: ([np.ndarray], [np.ndarray], [np.ndarray], [np.ndarray]), seed: int):
        random.seed(seed)
        self.pop_size = popSize
        self.pc = pc
        self.pm = pm
        self.tournament_size = tournament_size
        self.trainInputs = nn_data[0]
        self.trainOutputs = nn_data[1]
        self.testInputs = nn_data[2]
        self.testOutputs = nn_data[3]
        input_size = self.testInputs[0].shape[0]
        output_size = self.testOutputs[0].shape[0]
        self.population = generate_population(self.hrange, startPopSize, input_size=input_size, output_size=output_size)

        self.supervisor.get_algo_data(pm=self.pm, pc=self.pc, ts=self.tournament_size, sps=startPopSize,
                                      ps=self.pop_size, fracs=self.fractions, hrange=self.hrange)

    def run(self, iterations: int, power: int = 1) -> AnnPoint:
        pool = mp.Pool(power)
        self.supervisor.start(iterations)

        print("start")

        for i in range(iterations):
            eval_pop = self.calculate_fitnesses(pool, self.population)

            self.supervisor.check_point(eval_pop, i)

            crossed = []

            while len(crossed) < self.pop_size:
                c1 = self.select(eval_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= self.pc:
                    c2 = self.select(eval_pop=eval_pop)
                    cr_result = self.crossover(c1, c2)
                    crossed.extend(cr_result)
                else:
                    crossed.append(c1)

            new_pop = []

            for ind in range(len(crossed)):
                new_pop.append(self.mutate(crossed[ind]))

            self.population = new_pop

        eval_pop = self.calculate_fitnesses(pool, self.population)
        pool.close()

        eval_pop_sorted = sorted(eval_pop, key=lambda x: x[1], reverse=True)

        return eval_pop_sorted[0][0]

    def select(self, eval_pop: [[AnnPoint, float]]):
        chosen = choose_without_repetition(options=eval_pop, count=self.tournament_size)
        chosen_sorted = sorted(chosen, key=lambda x: x[1], reverse=True)
        return chosen_sorted[0][0]

    def crossover(self, pointA: AnnPoint, pointB: AnnPoint) -> [AnnPoint]:
        pointA = pointA.copy()
        pointB = pointB.copy()

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.hiddenLayerCount
            pointA.hiddenLayerCount = pointB.hiddenLayerCount
            pointB.hiddenLayerCount = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.neuronCount
            pointA.neuronCount = pointB.neuronCount
            pointB.neuronCount = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.actFun.copy()
            pointA.actFun = pointB.actFun.copy()
            pointB.actFun = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.aggrFun.copy()
            pointA.aggrFun = pointB.aggrFun.copy()
            pointB.aggrFun = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.lossFun.copy()
            pointA.lossFun = pointB.lossFun.copy()
            pointB.lossFun = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.learningRate
            pointA.learningRate = pointB.learningRate
            pointB.learningRate = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.momCoeff
            pointA.momCoeff = pointB.momCoeff
            pointB.momCoeff = tmp

        sr = random.random()
        if sr < 0.5:
            tmp = pointA.batchSize
            pointA.batchSize = pointB.batchSize
            pointB.batchSize = tmp

        return [pointA, pointB]



    def mutate(self, point: AnnPoint) -> AnnPoint:
        point = point.copy()

        mr = random.random()
        if mr < self.pm:
            point.hiddenLayerCount = try_choose_different(point.hiddenLayerCount, range(self.hrange.layerCountMin, self.hrange.layerCountMax + 1))

        mr = random.random()
        if mr < self.pm:
            point.neuronCount = random.uniform(self.hrange.neuronCountMin, self.hrange.neuronCountMax)

        mr = random.random()
        if mr < self.pm:
            point.actFun = try_choose_different(point.actFun, self.hrange.actFunSet)

        mr = random.random()
        if mr < self.pm:
            point.aggrFun = try_choose_different(point.aggrFun, self.hrange.aggrFunSet)

        mr = random.random()
        if mr < self.pm:
            point.lossFun = try_choose_different(point.lossFun, self.hrange.lossFunSet)

        mr = random.random()
        if mr < self.pm:
            point.learningRate = random.uniform(self.hrange.learningRateMin, self.hrange.learningRateMax)

        # TODO radzenie sobie z błedami w obliczeniach
        mr = random.random()
        if mr < self.pm:
            point.momCoeff = random.uniform(self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax)

        mr = random.random()
        if mr < self.pm:
            point.batchSize = random.uniform(self.hrange.batchSizeMin, self.hrange.batchSizeMax)

        return point

    def calculate_fitnesses(self, pool: mp.Pool, points: [AnnPoint]) -> [[AnnPoint, float]]:
        count = len(points)

        touches = [0] * count

        estimates = [[point, 0] for point in points]

        for f in range(len(self.fractions)):
            frac = self.fractions[f]

            estimates = sorted(estimates, key=lambda x: x[1], reverse=True)
            comp_count = ceil(frac * count)
            to_compute = [est[0] for est in estimates[0:comp_count]]
            seeds = [random.randint(0, 1000) for i in range(len(to_compute))]

            # new_fitnesses = [self.calculate_fitness(to_compute[i], seeds[i])for i in range(len(to_compute))]

            estimating_async_results = [pool.apply_async(func=self.calculate_fitness, args=(to_compute[i], seeds[i])) for i in range(len(to_compute))]
            [estimation_result.wait() for estimation_result in estimating_async_results]
            new_fitnesses = [result.get() for result in estimating_async_results]

            ori = 1

            for i in range(comp_count):
                new_fit = new_fitnesses[i]
                curr_est = estimates[i][1]
                touch = touches[i]

                curr_sum = curr_est * touch
                new_sum = curr_sum + new_fit
                new_est = new_sum / (touch + 1)

                estimates[i][1] = new_est
                touches[i] += 1

        return estimates

    def calculate_fitness(self, point: AnnPoint, seed: int):
        network = network_from_point(point, seed)

        results = []

        for i in range(self.learningIts):
            network.train(inputs=self.trainInputs, outputs=self.trainOutputs, epochs=1)
            test_results = network.test(test_input=self.testInputs, test_output=self.testOutputs)
            result = mean(test_results[0:3])
            results.append(result)

        y = np.array(results)
        x = np.array(list(range(0, self.learningIts)))

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        reg = LinearRegression().fit(x, y)
        slope = reg.coef_

        return results[-1] * punishment_function(slope)



