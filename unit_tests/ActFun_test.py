import pytest
from ann_point.Functions import *

def test_relu():
    arg = np.array([[9], [1], [0], [-1]])
    exp = np.array([[9], [1], [0], [0]])

    assert np.array_equal(exp, ReLu().compute(arg))

def test_relu_der():
    arg = np.array([[9], [0], [-1]])
    exp = np.array([[1], [0], [0]])

    assert np.array_equal(exp, ReLu().computeDer(arg))

def test_sigma():
    arg = np.array([[0], [-1], [2]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.compute(arg), np.array([[0.5], [0.268941421], [0.880797077]]), atol=1e-6))

def test_sigma_der():
    arg = np.array([[0], [-1], [2]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.computeDer(arg), np.array([[0.25], [0.196611933], [0.104993585]]), atol=1e-6))

def test_softmax():
    arg = np.array([[1], [2], [0.5]])

    sm = Softmax()

    assert np.all(np.isclose(sm.compute(arg), np.array([[0.23], [0.63], [0.14]]), atol=1e-2))