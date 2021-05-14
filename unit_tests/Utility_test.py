from utility.Utility import *
import pytest

def test_batch_divide_round():
    inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
    outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14]), np.array([15])]

    batches = divideIntoBatches(inputs, outputs, 2)

    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 2

    assert np.array_equal(batches[0][0][0], np.array([0]))
    assert np.array_equal(batches[0][0][1], np.array([10]))
    assert np.array_equal(batches[0][1][0], np.array([1]))
    assert np.array_equal(batches[0][1][1], np.array([11]))

    assert np.array_equal(batches[1][0][0], np.array([2]))
    assert np.array_equal(batches[1][0][1], np.array([12]))
    assert np.array_equal(batches[1][1][0], np.array([3]))
    assert np.array_equal(batches[1][1][1], np.array([13]))

    assert np.array_equal(batches[2][0][0], np.array([4]))
    assert np.array_equal(batches[2][0][1], np.array([14]))
    assert np.array_equal(batches[2][1][0], np.array([5]))
    assert np.array_equal(batches[2][1][1], np.array([15]))



def test_batch_divide_not_round():
    inputs = [np.array([0]), np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    outputs = [np.array([10]), np.array([11]), np.array([12]), np.array([13]), np.array([14])]

    batches = divideIntoBatches(inputs, outputs, 2)

    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1

    assert np.array_equal(batches[0][0][0], np.array([0]))
    assert np.array_equal(batches[0][0][1], np.array([10]))
    assert np.array_equal(batches[0][1][0], np.array([1]))
    assert np.array_equal(batches[0][1][1], np.array([11]))

    assert np.array_equal(batches[1][0][0], np.array([2]))
    assert np.array_equal(batches[1][0][1], np.array([12]))
    assert np.array_equal(batches[1][1][0], np.array([3]))
    assert np.array_equal(batches[1][1][1], np.array([13]))

    assert np.array_equal(batches[2][0][0], np.array([4]))
    assert np.array_equal(batches[2][0][1], np.array([14]))


# test_batch_divide_round()
