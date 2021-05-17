import warnings

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

class TanH(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        up = np.exp(arg) - np.exp(-arg)
        low = np.exp(arg) + np.exp(-arg)

        low[np.where(np.isposinf(up))] = 1
        up[np.where(np.isposinf(up))] = 1
        low[np.where(np.isneginf(up))] = 1
        up[np.where(np.isneginf(up))] = -1

        result = up / low

        return result

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        low = 2 / (np.exp(arg) + np.exp(-arg))
        return low ** 2

    def copy(self):
        return TanH()

    def to_string(self):
        return "TH"


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
        arg_c = arg - np.max(arg)
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
        return np.mean(np.square(res - corr), axis=0)[0]

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        return 2 * (res - corr) / res.shape[0]

    def copy(self):
        return QuadDiff()

    def to_string(self):
        return "QD"

class CrossEntropy(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        result = np.sum(np.multiply(corr, np.log(res + 1e-15)), axis=0)[0]
        return result

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        return -corr / (res + 1e-15)

    def copy(self):
        return CrossEntropy()

    def to_string(self):
        return "CE"




# TODO cross entropy
# TODO softmax
# TODO tanh? other acts and losses

















