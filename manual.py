import random

from TupleForTest import TupleForTest
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from evolving_classifier.FitnessCalculator import CNFitnessCalculator
from evolving_classifier.FitnessFunction import CNFF
from evolving_classifier.operators.Rejects.FinalCO1 import FinalCO1
from evolving_classifier.operators.MutationOperators import FinalMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection
from tester import run_tests
from utility.Utility import generate_counting_problem, get_default_hrange_ga
import os


if __name__ == "__main__":
    def run_ec():
        ec = EvolvingClassifier()
        x, y = generate_counting_problem(10, 3)
        X, Y = generate_counting_problem(10, 3)
        popSize = 30
        iterations = 30
        data = (x, y, X, Y)
        seed = 1001
        hrange = get_default_hrange_ga()
        hrange.max_hidden = 5
        hrange.max_it = 2
        ct = FinalCO1
        mt = FinalMutationOperator
        st = [TournamentSelection, 2]
        fft = [CNFF]
        fct = CNFitnessCalculator

        tests = []
        tests.append(TupleForTest(name="iris_01", rep=2, seed=seed, popSize=popSize,
                                  data=data, iterations=iterations, hrange=hrange,
                                  ct=ct, mt=mt, st=st,
                                  fft=fft, fct=fct, reg=False))

        rh = run_tests(tts=tests, directory_for_tests=f"algo_tests{os.path.sep}to_delete{os.path.sep}manual", power=12)

        random.seed(seed)
        seed1 = random.randint(0, 10**6)
        seed2 = random.randint(0, 10**6)
        ec.prepare(popSize=popSize, nn_data=data, seed=seed1, hrange=hrange, ct=ct, mt=mt, st=st, fft=fft, fct=fct)
        net = ec.run(iterations=iterations, power=12)

        ec = EvolvingClassifier()
        ec.prepare(popSize=popSize, nn_data=data, seed=seed2, hrange=hrange, ct=ct, mt=mt, st=st, fft=fft, fct=fct)
        net2 = ec.run(iterations=iterations, power=12)

        ori = 1

    run_ec()
