from evolving_classifier.EvolvingClassifier import *
from evolving_classifier.operators.MutationOperators import *
from evolving_classifier.operators.SelectionOperator import *


def test_determinism():
    random.seed(1001)
    np.random.seed(1001)

    ec = EvolvingClassifier()

    count = 10
    size = 5
    x,y = generate_counting_problem(count, size)
    X,Y = generate_counting_problem(ceil(count), size)

    tests = []

    for i in range(5):
        ec.prepare(10, (x, y, X, Y), 1001)
        net = ec.run(10, 1)
        tests.append(net.test(X, Y))

    for i in range(len(tests) - 1):
        for j in range(i + 1, len(tests)):
            t1 = tests[i]
            t2 = tests[j]

            assert np.array_equal(t1[0], t2[0])

def test_sync_async_determinism():
    if __name__ == "unit_tests.heavy_tests.EC_test" or __name__ == "__main__":
        random.seed(1001)
        np.random.seed(1001)

        ec = EvolvingClassifier()

        count = 20
        size = 5
        x,y = generate_counting_problem(count, size)
        X,Y = generate_counting_problem(ceil(count), size)

        tests = []
        processes = [1, 2, 4, 8, 12]
        for i in range(5):
            ec.prepare(20, (x, y, X, Y), 1001)
            net = ec.run(20, processes[i])
            tests.append(net.test(X, Y))

        for i in range(len(tests) - 1):
            for j in range(i + 1, len(tests)):
                t1 = tests[i]
                t2 = tests[j]

                assert np.array_equal(t1[0], t2[0])



















# test_determinism()