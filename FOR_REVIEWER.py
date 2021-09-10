from TupleForTest import TupleForTest
from data_plotter import read_all_frames_from_directory, plot_min_max_avg
from evolving_classifier.FitnessFunction import MEFF
from evolving_classifier.LsmFitnessCalculator import LsmFitnessCalculator
from evolving_classifier.operators.LsmCrossoverOperator import LsmCrossoverOperator
from evolving_classifier.operators.LsmMutationOperator import LsmMutationOperator
from evolving_classifier.operators.SelectionOperator import TournamentSelection95
from tester import run_tests
from utility.Utility import get_doc_hrange_eff
import ts_iris
import ts_wines
import ts_german
import random
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    x, y, X, Y = ts_iris.get_data()

    tests = []

    repetitions = 5
    population_size = 200
    iterations = 50
    starg = 4
    power = 4
    seed = 1001

    directory = "iris_review"


    hrange = get_doc_hrange_eff()
    tests.append(TupleForTest(name=directory, rep=repetitions, seed=seed, popSize=population_size,
                              data=[x, y, X, Y], iterations=iterations, hrange=hrange,
                              ct=LsmCrossoverOperator, mt=LsmMutationOperator, st=[TournamentSelection95, starg],
                              fft=[MEFF], fct=LsmFitnessCalculator, reg=False))

    run_tests(tts=tests, directory_for_tests=f"review_tests", power=power)

    data_frames = read_all_frames_from_directory(f"review_tests{os.path.sep}{directory}")
    plot_min_max_avg(frames=data_frames, parameter_name="meff", title="Efektywność", xtitle="iteracje", ytitle="efektywność",
                     spath="review_eff")
    plot_min_max_avg(frames=data_frames, parameter_name="nc", title="Liczba neuronów", xtitle="iteracje", ytitle="liczba neuronów",
                     spath="review_eff")

    plt.show()

