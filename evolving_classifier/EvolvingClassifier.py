import math
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.FinalCO1 import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from utility.RunHistory import RunHistory
from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

# TODO - B - usunąć supervisora? Chyba nie jest używany
# TODO - B - czy sortowanie jest w ogóle koniecznie poza wypisywaniem?
class EvolvingClassifier:
    def __init__(self):

        self.hrange = None
        self.population = []
        self.pop_size = -1
        self.trainInputs = []
        self.trainOutputs = []

        self.co = None
        self.mo = None
        self.so = None
        self.ff = None
        self.fc = None

        self.history = None

    #TODO - C - ziarno nieobowiązkowe?
    def prepare(self, popSize: int, nn_data: ([np.ndarray], [np.ndarray]),
                seed: int, hrange: HyperparameterRange = None, ct: type = None, mt: type = None, st: [Any] = None, fft: [Any] = None,
                fct: type = None):
        if hrange is None:
            self.hrange = get_default_hrange_ga()
        else:
            self.hrange = hrange

        if ct == None:
            self.co = FinalCO1(self.hrange)
        else:
            self.co = ct(self.hrange)

        if mt == None:
            self.mo = FinalMutationOperator(self.hrange)
        else:
            self.mo = mt(self.hrange)

        if st == None:
            self.so = TournamentSelection(max(round(0.02 * popSize), 2))
        elif len(st) == 1:
            self.so = st[0]()
        else:
            self.so = st[0](st[1])

        if fft == None:
            self.ff = CNFF()
        elif len(fft) == 1:
            self.ff = fft[0]()
        else:
            self.ff = fft[0](fft[1]())

        if fct == None:
            self.fc = CNFitnessCalculator()
        else:
            self.fc = fct()

        random.seed(seed)
        np.random.seed(seed)

        self.pop_size = popSize
        self.trainInputs = nn_data[0]
        self.trainOutputs = nn_data[1]
        input_size = self.trainInputs[0].shape[0]
        output_size = self.trainOutputs[0].shape[0]
        self.population = generate_population(self.hrange, popSize, input_size=input_size, output_size=output_size)

        self.history = RunHistory()

    def run(self, iterations: int, power: int = 1, verbose: bool = False) -> ChaosNet:
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None

        best = [self.population[0], -math.inf]

        for i in range(iterations):
            eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                       trainOutputs=self.trainOutputs)

            self.history.add_it_hist(eval_pop)

            pc = 20
            lc = 300
            if verbose:
                if i % pc == 0:
                    print(f"{i + 1} - {eval_pop[0].ff} - {eval_pop[0].net.to_string()},")
            else:
                if i % lc == 0:
                    if i > 0:
                        print()
                    print("    ", end="")
                if i % pc == 0 or i == iterations - 1:
                    if i % lc != 0:
                        print(", ", end="")
                    print(f"{round((i + 1)/iterations * 100, 2)}%", end="")

            if eval_pop[0].ff >= best[1]:
                best = [eval_pop[0].net.copy(), eval_pop[0].ff]

            crossed = []

            while len(crossed) < self.pop_size:
                c1 = self.so.select(val_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= 10 ** c1.c_prob:
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

        eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                   trainOutputs=self.trainOutputs)

        if power > 1:
            pool.close()

        if eval_pop[0].ff >= best[1]:
            best = [eval_pop[0].net.copy(), eval_pop[0].ff]

        if not verbose:
            print()

        return best[0]

