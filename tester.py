import random
from statistics import mean
import numpy as np

np.seterr(all='ignore')

from evolving_classifier.EvolvingClassifier import *
from TupleForTest import TupleForTest
import numpy as np


def run_tests(tts: [TupleForTest], power: int) -> [[ChaosNet]]:
    resultss = []

    for tt in tts:
        results = []
        random.seed(tt.seed)
        np.random.seed(tt.seed)

        seeds = []
        for i in range(tt.rep):
            seeds.append(random.randint(0, 10**6))

        for i in range(tt.rep):
            ec = EvolvingClassifier()
            ec.prepare(popSize=tt.popSize, nn_data=tt.data, seed=seeds[i], hrange=tt.hrange, ct=tt.ct, mt=tt.mt,
                       st=tt.st, fft=tt.fft, fct=tt.fct, starg=tt.starg, fftarg=tt.fftarg)
            net = ec.run(iterations=tt.iterations, power=power)
            results.append(net.copy())
            print()
            if tt.reg == False:
                tr = net.test(tt.data[2], tt.data[3])
                print(mean(tr[:3]))
            else:
                tr = net.test(tt.data[2], tt.data[3], lf=tt.fftarg())
                print(tr[4])

            # print(tr[3])

        resultss.append(results)

    return resultss

if __name__ == '__main__':
    seed = 22223333
    random.seed(seed)
    np.random.seed(seed)

    count_tr = 500
    count_test = 500
    size = 7
    x,y = generate_counting_problem(count_tr, size)
    X,Y = generate_counting_problem(ceil(count_test), size)

    x,y = generate_square_problem(100, -1, 1)
    X,Y = generate_square_problem(100, -1, 1)

    hrange = HyperparameterRange((-1, 1), (-1, 1), (1, 10), (0, 20), [Poly2(), Poly3(), Identity(), ReLu(), Sigmoid(), TanH(), Softmax(), GaussAct(), LReLu(), SincAct()],
                                 mut_radius=(0.001, 1), wb_mut_prob=(0.001, 1), s_mut_prob=(0.001, 1), p_mutation_prob=(0.01, 1), c_prob=(0.2, 1),
                                 r_prob=(0, 1))

    test = TupleForTest(rep=1, seed=1001, popSize=300, data=[x, y, X, Y], iterations=300, hrange=hrange,
                        ct=FinalCrossoverOperator, mt=FinalMutationOperator, st=TournamentSelection,
                        fft=CNFF4, fct=CNFitnessCalculator, starg=0.05, fftarg=QuadDiff, reg=True)

    net = run_tests([test], 12)[0][0]

    args = [-1, -0.5, 0, 0.5, 1]

    for i in range(len(args)):
        print(net.run(np.array([[args[i]]])))




