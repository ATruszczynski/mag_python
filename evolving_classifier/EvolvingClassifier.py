from evolving_classifier.LsmFitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.LsmCrossoverOperator import LsmCrossoverOperator
from evolving_classifier.operators.Rejects.FinalCO1 import *
from evolving_classifier.operators.LsmMutationOperator import *
from evolving_classifier.operators.SelectionOperator import *
from utility.RunHistory import RunHistory
from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

tol = 20

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

    def prepare(self, popSize: int, nn_data: ([np.ndarray], [np.ndarray]),
                seed: int, hrange: HyperparameterRange = None, ct: type = None, mt: type = None, st: [Any] = None, fft: [Any] = None,
                fct: type = None):
        if hrange is None:
            self.hrange = get_default_hrange_ga()
        else:
            self.hrange = hrange

        if ct == None:
            self.co = LsmCrossoverOperator(self.hrange)
        else:
            self.co = ct(self.hrange)

        if mt == None:
            self.mo = LsmMutationOperator(self.hrange)
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
            self.fc = LsmFitnessCalculator()
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

    def run(self, iterations: int, power: int = 1, verbose: bool = False) -> LsmNetwork:
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None

        fake = CNDataPoint(self.population[0])
        fake.add_data(new_ff=[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf], new_conf_mat=np.zeros((0, 0)))
        bests = [fake.copy()]

        for i in range(len(self.population)):
            self.population[i].conf_mat = np.ones((1, 1))

        for i in range(iterations):
            eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                       trainOutputs=self.trainOutputs)


            self.history.add_it_hist(eval_pop)

            curr_it_best = eval_pop[0]
            if are_ffs_ge(curr_it_best, bests[-1]):
                bests.append(curr_it_best.copy())
            else:
                bests.append(bests[-1].copy())

            if len(bests) >= tol:
                if are_ffs_eq(bests[0], bests[-1]):
                    break
                else:
                    bests = bests[1:]

            pc = 1
            lc = pc * 8
            if verbose:
                if i % pc == 0:
                    print(f"{i + 1} - {eval_pop[0].ff[0]} - {eval_pop[0].to_string()},")
            else:
                if i % lc == 0:
                    if i > 0:
                        print()
                    print("    ", end="")
                if i % pc == 0 or i == iterations - 1:
                    if i % lc != 0:
                        print(", ", end="")
                    print(f"{i} - ({round((i + 1)/iterations * 100, 2)}%) - {round(bests[-1].ff[0], 4)}", end="")


            crossed = []

            while len(crossed) < self.pop_size:
                c1 = self.so.select(val_pop=eval_pop)
                cr = random.random()

                if len(crossed) <= self.pop_size - 2 and cr <= 10 ** c1.c_prob:
                    c2 = self.so.select(val_pop=eval_pop)
                    cr_result = self.co.crossover(c1, c2)
                    crossed.extend(cr_result)
                else:
                    crossed.append(c1.copy())

            new_pop = []

            for ind in range(len(crossed)):
                to_mutate = crossed[ind]
                new_pop.append(self.mo.mutate(to_mutate))

            self.population = new_pop

        eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                   trainOutputs=self.trainOutputs)

        if power > 1:
            pool.close()

        if are_ffs_ge(eval_pop[0], bests[-1]):
            bests.append(eval_pop[0].copy())

        if not verbose:
            print()

        return bests[-1].net

