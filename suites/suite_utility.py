from TupleForTest import TupleForTest, assert_tts_same
from tester import run_tests
import os

directory_for_tests=f"..{os.path.sep}algo_tests"
trash_can=f"..{os.path.sep}algo_tests{os.path.sep}to_delete"


# TODO - S - test
def try_check_if_all_tests_computable(tests: [TupleForTest], directory, power: int):
    for i in range(len(tests) - 1):
        for j in range(i + 1, len(tests)):
            t1 = tests[i].copy()
            t2 = tests[j].copy()
            assert t1.name != t2.name

            t1.name = ""
            t2.name = ""

            try:
                assert_tts_same(t1, t2)
            except AssertionError:
                pass
            else:
                assert False

    ctest = []
    for i in range(len(tests)):
        t = tests[i].copy()
        t.name = "ctt_" + t.name
        t.rep = 1
        t.popSize = 20
        t.iterations = 5
        ctest.append(t)

    run_tests(ctest, directory, power)


