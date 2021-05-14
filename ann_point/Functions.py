import numpy as np

class ActFun:
    def compute(self, arg: np.ndarray) -> np.ndarray:
        pass

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        pass

    def copy(self):
        return ActFun()

    def to_string(self):
        pass

class ReLu(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg = arg.copy()
        return np.maximum(0, arg)

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        result = arg.copy()
        result[arg > 0] = 1
        result[arg <= 0] = 0
        return result

    def copy(self):
        return ReLu()

    def to_string(self):
        return "RL"


class Sigmoid(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-arg))

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        return self.compute(arg) * (1 - self.compute(arg))

    def copy(self):
        return Sigmoid()

    def to_string(self):
        return "SG"


class Softmax(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg_c = arg - np.min(arg)
        exp_a = np.exp(arg_c)
        return exp_a / exp_a.sum(axis=0, keepdims=True)

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        return self.compute(arg) * (1 - self.compute(arg)) # TODO something may be wrong here

    def copy(self):
        return Softmax()

    def to_string(self):
        return "SM"














class LossFun:
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        pass

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        pass

    def copy(self):
        return ActFun()

    def to_string(self):
        pass


class QuadDiff(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        return np.sum(np.square(res - corr), axis=0)[0]

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        return 2 * (res - corr)

    def copy(self):
        return QuadDiff()

    def to_string(self):
        return "QD"


# TODO cross entropy
# TODO softmax
# TODO tanh? other acts and losses

















