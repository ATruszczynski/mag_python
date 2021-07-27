import numpy as np

eps = 1e-10

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
        result = np.zeros((arg.shape[0], arg.shape[0]))
        diag = arg.copy()
        diag[arg > 0] = 1
        diag[arg <= 0] = 0
        np.fill_diagonal(result, diag)

        return result

    def copy(self):
        return ReLu()

    def to_string(self):
        return "RL"


class LReLu(ActFun):
    def __init__(self, a: float=0.01):
        self.a = a

    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg = arg.copy()
        return np.maximum(self.a * arg, arg)

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        result = np.zeros((arg.shape[0], arg.shape[0]))
        diag = arg.copy()
        diag[arg > 0] = 1
        diag[arg <= 0] = self.a
        np.fill_diagonal(result, diag)

        return result

    def copy(self):
        return LReLu(self.a)

    def to_string(self):
        return f"L{self.a}"


class GaussAct(ActFun):
    def __init__(self):
        pass

    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg = arg.copy()
        return np.exp(-arg ** 2)

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        result = np.zeros((arg.shape[0], arg.shape[0]))
        diag = -2 * arg * np.exp(-arg ** 2)
        np.fill_diagonal(result, diag)
        return result

    def copy(self):
        return GaussAct()

    def to_string(self):
        return "GS"

class SincAct(ActFun):
    def __init__(self):
        pass

    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg = arg.copy()
        res = np.zeros(arg.shape)
        res[arg == 0] = 1
        res[arg != 0] = np.sin(arg[arg != 0]) / (arg[arg != 0] + eps)
        return res

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        result = np.zeros((arg.shape[0], arg.shape[0]))
        diag = np.zeros(arg.shape)
        diag[arg == 0] = 0
        diag[arg != 0] = (np.cos(arg[arg != 0]) / (arg[arg != 0] + eps)) - (np.sin(arg[arg != 0]) / (arg[arg != 0] ** 2 + eps))
        np.fill_diagonal(result, diag)
        return result

    def copy(self):
        return SincAct()

    def to_string(self):
        return "SC"

class TanH(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        up = np.exp(arg) - np.exp(-arg)
        low = np.exp(arg) + np.exp(-arg)

        if not isinstance(up, np.ndarray):
            up = np.array([[up]])
        if not isinstance(low, np.ndarray):
            low = np.array([[low]])

        low[np.where(np.isposinf(up))] = 1
        up[np.where(np.isposinf(up))] = 1
        low[np.where(np.isneginf(up))] = 1
        up[np.where(np.isneginf(up))] = -1

        result = up / (low + eps)

        return result

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        diag = 2 / (np.exp(arg) + np.exp(-arg))
        # diqg = 1 - self.compute(arg) ** 2
        diag = diag ** 2 #TODO is this der correct?
        return np.diagflat(diag)

    def copy(self):
        return TanH()

    def to_string(self):
        return "TH"


class Sigmoid(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-arg))

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        com = self.compute(arg)
        diag = com * (1 - com)
        return np.diagflat(diag)

    def copy(self):
        return Sigmoid()

    def to_string(self):
        return "SG"


class Softmax(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        arg_c = arg - np.max(arg)
        exp_a = np.exp(arg_c)
        return exp_a / (exp_a.sum(axis=0, keepdims=True) + eps)

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        diag = self.compute(arg)
        result = np.diagflat(diag)

        result = result - np.dot(diag, diag.T)

        return result

    def copy(self):
        return Softmax()

    def to_string(self):
        return "SM"

class Identity(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        return arg

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        pass

    def copy(self):
        return Identity()

    def to_string(self):
        return "ID"


class Poly2(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        return arg ** 2

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        pass

    def copy(self):
        return Poly2()

    def to_string(self):
        return "P2"


class Poly3(ActFun):
    def compute(self, arg: np.ndarray) -> np.ndarray:
        return arg ** 3

    def computeDer(self, arg: np.ndarray) -> np.ndarray:
        pass

    def copy(self):
        return Poly3()

    def to_string(self):
        return "P3"














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

class MeanDiff(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        return np.mean(np.abs(res - corr), axis=0)[0]

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        return np.sign(res - corr) / res.shape[0]

    def copy(self):
        return MeanDiff()

    def to_string(self):
        return "MAE"

class CrossEntropy(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        result = np.sum(np.multiply(corr, np.log(res + 1e-15)), axis=0)[0]
        return -result

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        return -corr / (res + 1e-15)

    def copy(self):
        return CrossEntropy()

    def to_string(self):
        return "CE"

class ChebyshevLoss(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        result = np.max(np.abs(res - corr), axis=0)[0]
        return result

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        diffs = np.abs(res - corr)
        highest = np.argmax(diffs)
        signs = np.sign(res - corr)
        result = np.zeros(res.shape)
        result[highest] = signs[highest]
        return result

    def copy(self):
        return ChebyshevLoss()

    def to_string(self):
        return "CL"

#TODO test
class QuasiCrossEntropy(LossFun):
    def compute(self, res: np.ndarray, corr: np.ndarray) -> float:
        result = np.sum(np.multiply(corr, np.abs(res - corr)))
        return result

    def computeDer(self, res: np.ndarray, corr: np.ndarray) -> np.ndarray:
        raise Exception()

    def copy(self):
        return QuasiCrossEntropy()

    def to_string(self):
        return "QCE"




















