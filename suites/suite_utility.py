from TupleForTest import TupleForTest
from tester import run_tests
import os

directory_for_tests=f"..{os.path.sep}algo_test"

# TODO - S - test
def try_check_if_all_tests_computable(tests: [TupleForTest], power: int):
    for i in range(len(tests) - 1):
        for j in range(i + 1, len(tests)):
            assert tests[i].name != tests[j].name

    ctest = []
    for i in range(len(tests)):
        t = tests[i].copy()
        t.name = "ctt_" + t.name
        t.rep = 1
        t.popSize = 20
        t.iterations = 5
        ctest.append(t)

    run_tests(ctest, power)


