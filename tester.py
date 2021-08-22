import random
from statistics import mean
from sklearn import datasets

from evolving_classifier.operators.PuzzleCO import PuzzleCO
from evolving_classifier.operators.PuzzleCO2 import PuzzleCO2
from evolving_classifier.operators.CrossoverOperator2 import FinalCrossoverOperator2
from evolving_classifier.operators.CrossoverOperator3 import FinalCrossoverOperator3
from evolving_classifier.operators.CrossoverOperator4 import FinalCrossoverOperator4
from evolving_classifier.operators.CrossoverOperator5 import FinalCrossoverOperator5
from evolving_classifier.EvolvingClassifier import *
from TupleForTest import TupleForTest
import numpy as np
import os.path

np.seterr(all='ignore')

directory_for_tests="algo_tests"

#TODO - S - add some summary report file per test
# TODO - S - check if everythoing works correctly

def run_tests(tts: [TupleForTest], power: int) -> [[ChaosNet]]:
    resultss = []

    if not os.path.exists(directory_for_tests):
        os.mkdir(directory_for_tests)

    for tt in tts:
        subdir_name = f"{directory_for_tests}{os.path.sep}{tt.name}"
        if not os.path.exists(subdir_name):
            os.mkdir(subdir_name)

        fpath = f"{directory_for_tests}{os.path.sep}{tt.name}{os.path.sep}{tt.name}-"

        create_test_data_file(fpath, tt)

        results = []
        random.seed(tt.seed)
        np.random.seed(tt.seed)

        seeds = []
        for i in range(tt.rep):
            seeds.append(random.randint(0, 10**6))

        for i in range(tt.rep):
            ec = EvolvingClassifier()
            ec.prepare(popSize=tt.popSize, nn_data=tt.data, seed=seeds[i], hrange=tt.hrange, ct=tt.ct, mt=tt.mt,
                       st=tt.st, fft=tt.fft, fct=tt.fct)
            net = ec.run(iterations=tt.iterations, power=power)
            net.net_to_file(fpath=fpath+f"best_{i + 1}.txt")
            ec.history.to_csv_file(fpath=fpath+f"rep_{i + 1}.csv", reg=tt.reg)
            results.append(net.copy())
            print()
            if tt.reg == False:
                tr = net.test(tt.data[2], tt.data[3])
                print(efficiency(tr[0]))
            else:
                tr = net.test(tt.data[2], tt.data[3], lf=tt.fftarg())
                print(tr[1])

        resultss.append(results)

    return resultss

def create_test_data_file(fpath: str, tt: TupleForTest):
    data_file = open(fpath + "data_file.txt", "w")

    data_file.write(f"test-data:\n")

    data_file.write(f"name: {tt.name} \n")
    data_file.write(f"rep: {tt.rep} \n")
    data_file.write(f"seed: {tt.seed} \n")
    data_file.write(f"popSize: {tt.popSize} \n")
    data_file.write(f"iterations: {tt.iterations} \n")
    data_file.write(f"ct: {tt.ct.__name__} \n")
    data_file.write(f"mt: {tt.mt.__name__} \n")
    data_file.write(f"st: {tt.st[0].__name__} \n")
    if len(tt.st) == 2:
        data_file.write(f"starg: {tt.st[1]} \n")
    data_file.write(f"fft: {tt.fft[0].__name__} \n")
    if len(tt.fft) == 2:
        data_file.write(f"fftarg: {tt.fft[1].__name__} \n")
    data_file.write(f"fct: {tt.fct.__name__} \n")
    data_file.write(f"reg: {tt.reg} \n")
    data_file.write(f"data_len: {len(tt.data[0])} \n")


    data_file.write(f"\nhyperparameters:\n")

    hrange = tt.hrange
    data_file.write(f"min_init_wei: {hrange.min_init_wei}\n")
    data_file.write(f"max_init_wei: {hrange.max_init_wei}\n")
    data_file.write(f"min_init_bia: {hrange.min_init_bia}\n")
    data_file.write(f"max_init_bia: {hrange.max_init_bia}\n")
    data_file.write(f"min_it: {hrange.min_it}\n")
    data_file.write(f"max_it: {hrange.max_it}\n")
    data_file.write(f"min_hidden: {hrange.min_hidden}\n")
    data_file.write(f"max_hidden: {hrange.max_hidden}\n")


    data_file.write(f"min_mut_radius: {hrange.min_mut_radius}\n")
    data_file.write(f"max_mut_radius: {hrange.max_mut_radius}\n")
    data_file.write(f"min_wb_mut_prob: {hrange.min_sqr_mut_prob}\n")
    data_file.write(f"max_wb_mut_prob: {hrange.max_sqr_mut_prob}\n")
    data_file.write(f"min_s_mut_prob: {hrange.min_lin_mut_prob}\n")
    data_file.write(f"max_s_mut_prob: {hrange.max_lin_mut_prob}\n")
    data_file.write(f"min_p_mut_prob: {hrange.min_p_mut_prob}\n")
    data_file.write(f"max_p_mut_prob: {hrange.max_p_mut_prob}\n")
    data_file.write(f"min_c_prob: {hrange.min_c_prob}\n")
    data_file.write(f"max_c_prob: {hrange.max_c_prob}\n")
    data_file.write(f"min_r_prob: {hrange.min_dstr_mut_prob}\n")
    data_file.write(f"max_r_prob: {hrange.max_dstr_mut_prob}\n")

    data_file.write("actfuns: ")
    for i in range(len(hrange.actFunSet)):
        data_file.write(hrange.actFunSet[i].to_string() + ", ")

    data_file.close()

    pass
# TODO - S - można zapisywać też inne statystyki sieci imo
if __name__ == '__main__':
    seed = 22223333
    random.seed(seed)
    np.random.seed(seed)

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x = [x.reshape((4, 1)) for x in x]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    xx = [x[i] for i in perm]
    yy = [y[i] for i in perm]

    x = [xx[i] for i in range(125)]
    y = [yy[i] for i in range(125)]
    X = [xx[i] for i in range(125, 150)]
    Y = [yy[i] for i in range(125, 150)]

    # count_tr = 500
    # count_test = 500
    # size = 5
    # x,y = generate_counting_problem(count_tr, size)
    # X,Y = generate_counting_problem(ceil(count_test), size)
    #
    # x,y = generate_square_problem(200, -5, 5)
    # X,Y = generate_square_problem(200, -5, 5)

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                 mut_radius=(0.001, 1), sqr_mut_prob=(0.001, 1), lin_mut_prob=(0.001, 1), p_mutation_prob=(0.01, 1), c_prob=(0.2, 1),
                                 dstr_mut_prob=(0, 1))

    minrr = -2
    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Identity(), ReLu(), Sigmoid(), Poly2(), Poly3(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                 mut_radius=(minrr, 0), sqr_mut_prob=(minrr, 0), lin_mut_prob=(minrr, 0), p_mutation_prob=(minrr, 0), c_prob=(-1, 0),
                                 dstr_mut_prob=(minrr, 0))

    # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 10), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
    #                              mut_radius=(minrr, minrr), wb_mut_prob=(minrr, minrr), s_mut_prob=(minrr, minrr), p_mutation_prob=(minrr, minrr), c_prob=(-0.12, -0.12),
    #                              r_prob=(minrr, minrr))

    # hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
    #                              mut_radius=(0.1, 0.1), wb_mut_prob=(0.01, 0.01), s_mut_prob=(0.01, 0.01), p_mutation_prob=(0.00, 0.00), c_prob=(0.7, 0.7),
    #                              r_prob=(0.01, 0.01))

    # test = TupleForTest(name="desu", rep=3, seed=1001, popSize=100, data=[x, y, X, Y], iterations=30, hrange=hrange,
    #                     ct=FinalCrossoverOperator, mt=FinalMutationOperator, st=TournamentSelection,
    #                     fft=CNFF4, fct=CNFitnessCalculator, starg=0.05, fftarg=QuadDiff, reg=True)
    # test2 = TupleForTest(name="desu2", rep=5, seed=1001, popSize=20, data=[x, y, X, Y], iterations=100, hrange=hrange,
    #                     ct=FinalCrossoverOperator, mt=FinalMutationOperator, st=TournamentSelection,
    #                     fft=CNFF4, fct=CNFitnessCalculator, starg=0.05, fftarg=QuadDiff, reg=True)
    # test = TupleForTest(name="desu3", rep=3, seed=1001, popSize=20, data=[x, y, X, Y], iterations=20, hrange=hrange,
    #                      ct=FinalCrossoverOperator, mt=FinalMutationOperator, st=TournamentSelection,
    #                      fft=CNFF4, fct=CNFitnessCalculator, starg=0.05, fftarg=QuadDiff, reg=True)
    tests = []
    pops = 20
    its = 20
    rep = 1
    seed = 12121212
    power = 1
    starg = 6
    # tests.append(TupleForTest(name="test_0", rep=1, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=PuzzleCO, mt=FinalMutationOperator, st=[TournamentSelection, starg],
    #                           fft=[CNFF], fct=CNFitnessCalculator, reg=False))
    tests.append(TupleForTest(name="test_00", rep=1, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
                              ct=FinalCrossoverOperator, mt=FinalMutationOperator, st=[TournamentSelectionSized, starg],
                              fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
    # tests.append(TupleForTest(name="test_1", rep=2, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=FinalCrossoverOperator2, mt=FinalMutationOperator, st=[TournamentSelection, starg],
    #                           fft=[CNFF4, QuadDiff], fct=CNFitnessCalculator, reg=False))
    # tests.append(TupleForTest(name="test_2", rep=rep, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=FinalCrossoverOperator3, mt=FinalMutationOperator, st=TournamentSelection,
    #                           fft=CNFF4, fct=CNFitnessCalculator, starg=starg, fftarg=QuadDiff, reg=False))
    # tests.append(TupleForTest(name="test_3", rep=rep, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=FinalCrossoverOperator4, mt=FinalMutationOperator, st=TournamentSelection,
    #                           fft=CNFF4, fct=CNFitnessCalculator, starg=starg, fftarg=QuadDiff, reg=False))
    # tests.append(TupleForTest(name="test_4", rep=rep, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=FinalCrossoverOperator5, mt=FinalMutationOperator, st=TournamentSelection,
    #                           fft=CNFF4, fct=CNFitnessCalculator, starg=starg, fftarg=QuadDiff, reg=False))
    # tests.append(TupleForTest(name="test_5", rep=rep, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=CO_Puzzle, mt=FinalMutationOperator, st=TournamentSelection,
    #                           fft=CNFF4, fct=CNFitnessCalculator, starg=starg, fftarg=QuadDiff, reg=False))
    # tests.append(TupleForTest(name="test_6", rep=rep, seed=seed, popSize=pops, data=[x, y, X, Y], iterations=its, hrange=hrange,
    #                           ct=PuzzleCO2, mt=FinalMutationOperator, st=TournamentSelection,
    #                           fft=CNFF4, fct=CNFitnessCalculator, starg=starg, fftarg=QuadDiff, reg=False))


    # net = run_tests([test, test2, test3], power=12)[0][0]
    resultsss = run_tests(tests, power=power)
    net = resultsss[0][0]
    print(net.to_string())

    args = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]



    print(net.links)
    print(net.weights)
    print(net.biases)
    print(net.to_string())
    res = net.test(X, Y)
    print(res[0])
    print(efficiency(res[0]))
    #
    # for i in range(len(args)):
    #     print(net.run(np.array([[args[i]]])))




