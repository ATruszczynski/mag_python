import math
from evolving_classifier.EC_supervisor import EC_supervisor
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

class EvolvingClassifier:
    def __init__(self, logs_path: str = ""):

        if logs_path == "":
            self.supervisor = EC_supervisor(logs)
        else:
            self.supervisor = EC_supervisor(logs_path)

        #TODO remove needless things from here
        self.population = []
        self.pop_size = -1
        self.trainInputs = []
        self.trainOutputs = []

        self.co = None
        self.mo = None
        self.so = None
        self.ff = None
        self.fc = None

    def prepare(self, popSize:int, nn_data: ([np.ndarray], [np.ndarray]),
                seed: int, hrange: HyperparameterRange = None):
        if hrange is None:
            self.hrange = get_default_hrange()
        else:
            self.hrange = hrange

        self.co = FinalCrossoverOperator(self.hrange)
        self.mo = FinalMutationOperator(self.hrange)
        self.so = TournamentSelection(0.01)
        self.ff = CNFF()
        self.fc = CNFitnessCalculator()

        random.seed(seed)
        np.random.seed(seed)

        self.pop_size = popSize
        self.trainInputs = nn_data[0]
        self.trainOutputs = nn_data[1]
        input_size = self.trainInputs[0].shape[0]
        output_size = self.trainOutputs[0].shape[0]
        self.population = generate_population(self.hrange, popSize, input_size=input_size, output_size=output_size)

    def run(self, iterations: int, power: int = 1) -> ChaosNet:
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None

        self.supervisor.start(iterations)

        input_size = self.trainInputs[0].shape[0]
        output_size = self.trainOutputs[0].shape[0]

        self.supervisor.get_algo_data(input_size=input_size, output_size=output_size, pmS=-666, pmE=-666,
                                      pcS=-666, pcE=-666,
                                      ts=-666, sps=len(self.population),
                                      ps=self.pop_size, fracs=[0], hrange=self.hrange, learningIts=self.ff.learningIts)

        self.supervisor.start(iterations=iterations)

        best = [self.population[0], -math.inf]

        for i in range(iterations):
            eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                       trainOutputs=self.trainOutputs)

            eval_pop = [eval_pop[i] for i in range(len(eval_pop)) if not np.isnan(eval_pop[i].ff)]

            sorted_eval = sorted(eval_pop, key=lambda x: x.ff, reverse=True)

            if sorted_eval[0].ff >= best[1]:
                best = [sorted_eval[0].net.copy(), sorted_eval[0].ff]

            self.supervisor.check_point(eval_pop, i)
            crossed = []

            while len(crossed) < self.pop_size:
                c1 = self.so.select(val_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= c1.c_prob:
                    c2 = self.so.select(val_pop=eval_pop)
                    cr_result = self.co.crossover(c1, c2)
                    crossed.extend(cr_result)
                else:
                    crossed.append(c1)

            new_pop = []

            for ind in range(len(crossed)):
                to_mutate = crossed[ind]
                new_pop.append(self.mo.mutate(to_mutate))

            self.population = new_pop
            print(f"best ff: {best[1]}")

        eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                   trainOutputs=self.trainOutputs)
        self.supervisor.check_point(eval_pop, iterations)
        if power > 1:
            pool.close()

        sorted_eval = sorted(eval_pop, key=lambda x: x.ff, reverse=True)
        if sorted_eval[0].ff >= best[1]:
            best = [sorted_eval[0].net.copy(), sorted_eval[0].ff]

        return best[0]

