import numpy as np

class MockCN:
    def __init__(self, val: float, mat: np.ndarray):
        self.val = val
        self.mat = mat.copy()


    def copy(self):
        return MockCN(self.val, self.mat.copy())

    def to_string(self):
        return f"{self.val} + {self.mat}"