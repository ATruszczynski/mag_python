from ann_point.AnnPoint import *
from ann_point.HyperparameterRange import *
import random

from utility.Utility import *


class EvolvingClassifier:
    def __init__(self, hrange: HyperparameterRange):
        self.hrange = hrange
        self.population = []
        self.fractions = []
        self.pop_size = -1
        self.pc = -1
        self.pm = -1
        self.tournament_size = 2

    def prepare(self, popSize: int, pc: float, pm: float, tournament_size: int, seed: int):
        random.seed(seed)

    def run(self, iterations: int) -> AnnPoint:

        for i in range(iterations):
            eval_pop = self.calculate_fitnesses(self.population)

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

        eval_pop = self.calculate_fitnesses(self.population)
        eval_pop_sorted = sorted(eval_pop, key=lambda x: x[1], reverse=True)

        return eval_pop_sorted[0]

    def select(self, eval_pop: [(AnnPoint, float)]):
        chosen = choose_without_repetition(options=eval_pop, count=self.tournament_size)
        chosen_sorted = sorted(chosen, key=lambda x: x[1], reverse=True)
        return chosen_sorted[0]

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

        mr = random.random()
        if mr < self.pm:
            point.momCoeff = random.uniform(self.hrange.momentumCoeffMin, self.hrange.momentumCoeffMax)

        return point

    def calculate_fitnesses(self, points: [AnnPoint]) -> [(AnnPoint, float)]:
        touches = [0] * len(points)

        fracs = [1]
        fracs.append(self.fractions)

        pass





