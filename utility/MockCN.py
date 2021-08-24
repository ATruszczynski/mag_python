import numpy as np

# TODO - C - useless file?
class MockCN:
    def __init__(self, neuron_count: float, mat: np.ndarray):
        self.neuron_count = neuron_count
        self.mat = mat.copy()


    def copy(self):
        return MockCN(self.neuron_count, self.mat.copy())

    def to_string(self):
        return f"{self.neuron_count} + {self.mat}"