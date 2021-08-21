import math
from evolving_classifier.EC_supervisor import EC_supervisor
from evolving_classifier.FitnessCalculator import *
from evolving_classifier.FitnessFunction import *
from evolving_classifier.operators.CrossoverOperator import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *
from utility.RunHistory import RunHistory
from utility.Utility import *

logs = "logs"
np.seterr(over='ignore')

# TODO - B - usunąć supervisora? Chyba nie jest używany
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
                seed: int, hrange: HyperparameterRange = None, ct: type = None, mt: type = None, st: type = None, fft: type = None,
                fct: type = None, starg: int = -1, fftarg: type = None):
        if hrange is None:
            self.hrange = get_default_hrange()
        else:
            self.hrange = hrange

        if ct == None:
            self.co = FinalCrossoverOperator(self.hrange)
        else:
            self.co = ct(self.hrange)

        if mt == None:
            self.mo = FinalMutationOperator(self.hrange)
        else:
            self.mo = mt(self.hrange)

        if st == None:
            self.so = TournamentSelection(0.01)
        elif starg is not None:
            self.so = st(starg)
        else:
            self.so = st()

        if fft == None:
            self.ff = CNFF()
        elif fftarg is not None:
            self.ff = fft(fftarg())
        else:
            self.ff = fft()

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
    #TODO - S - ma zwracać najlepszą znalezioną, czy najlepszą z ostatniej generacji?
    def run(self, iterations: int, power: int = 1) -> ChaosNet:
        #TODO - S - upewnić się, że wszędzie gdzie potrzebne sieci są posortowane
        if power > 1:
            pool = mp.Pool(power)
        else:
            pool = None

        best = [self.population[0], -math.inf]

        for i in range(iterations):
            eval_pop = self.fc.compute(pool=pool, to_compute=self.population, fitnessFunc=self.ff, trainInputs=self.trainInputs,
                                       trainOutputs=self.trainOutputs)

            eval_pop = [eval_pop[i] for i in range(len(eval_pop)) if not np.isnan(eval_pop[i].ff)]#TODO - S - moze się zdarzyć że to usunie wszystkie sieci
            if i % 10 == 0:
                print(f"{i + 1} - {eval_pop[0].ff} - {eval_pop[0].net.to_string()},")

            sorted_eval = sorted(eval_pop, key=lambda x: x.ff, reverse=True)
            self.history.add_it_hist(sorted_eval)

            if sorted_eval[0].ff >= best[1]:
                best = [sorted_eval[0].net.copy(), sorted_eval[0].ff]

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

        sorted_eval = sorted(eval_pop, key=lambda x: x.ff, reverse=True)
        if sorted_eval[0].ff >= best[1]:
            best = [sorted_eval[0].net.copy(), sorted_eval[0].ff]

        return best[0]

