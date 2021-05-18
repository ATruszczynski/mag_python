from evolving_classifier.EvolvingClassifier import *
import pytest

def test_crossover_operator():
    pointA = AnnPoint(5, 6, 2, 4, ReLu(), Sigmoid(), QuadDiff(), 1, 7, 3)
    pointB = AnnPoint(5, 6, 20, 40, TanH(), Softmax(), CrossEntropy(), 10, 70, 30)

    ec = EvolvingClassifier()

    res1 = ec.crossover(pointA, pointB, [0, 1, 0, 1, 0, 1, 1, 0])
    res2 = ec.crossover(pointA, pointB, [1, 0, 1, 0, 1, 1, 0, 1])

    pointC = res1[0]
    pointD = res1[1]
    pointE = res2[0]
    pointF = res2[1]

    points = [pointA, pointB, pointC, pointD, pointE, pointF]
    for i in range(0, len(points) - 1):
        for j in range(i + 1, len(points)):
            assert points[i].inputSize == points[j].inputSize
            assert points[i].outputSize == points[j].outputSize

    assert pointA.hiddenLayerCount == 2
    assert pointA.neuronCount == 4
    assert pointA.actFun.to_string() == ReLu().to_string()
    assert pointA.aggrFun.to_string() == Sigmoid().to_string()
    assert pointA.lossFun.to_string() == QuadDiff().to_string()
    assert pointA.learningRate == 1
    assert pointA.momCoeff == 7
    assert pointA.batchSize == 3

    assert pointB.hiddenLayerCount == 20
    assert pointB.neuronCount == 40
    assert pointB.actFun.to_string() == TanH().to_string()
    assert pointB.aggrFun.to_string() == Softmax().to_string()
    assert pointB.lossFun.to_string() == CrossEntropy().to_string()
    assert pointB.learningRate == 10
    assert pointB.momCoeff == 70
    assert pointB.batchSize == 30

    assert pointC.hiddenLayerCount == 20
    assert pointC.neuronCount == 4
    assert pointC.actFun.to_string() == TanH().to_string()
    assert pointC.aggrFun.to_string() == Sigmoid().to_string()
    assert pointC.lossFun.to_string() == CrossEntropy().to_string()
    assert pointC.learningRate == 1
    assert pointC.momCoeff == 7
    assert pointC.batchSize == 30

    assert pointD.hiddenLayerCount == 2
    assert pointD.neuronCount == 40
    assert pointD.actFun.to_string() == ReLu().to_string()
    assert pointD.aggrFun.to_string() == Softmax().to_string()
    assert pointD.lossFun.to_string() == QuadDiff().to_string()
    assert pointD.learningRate == 10
    assert pointD.momCoeff == 70
    assert pointD.batchSize == 3

    assert pointE.hiddenLayerCount == 2
    assert pointE.neuronCount == 40
    assert pointE.actFun.to_string() == ReLu().to_string()
    assert pointE.aggrFun.to_string() == Softmax().to_string()
    assert pointE.lossFun.to_string() == QuadDiff().to_string()
    assert pointE.learningRate == 1
    assert pointE.momCoeff == 70
    assert pointE.batchSize == 3

    assert pointF.hiddenLayerCount == 20
    assert pointF.neuronCount == 4
    assert pointF.actFun.to_string() == TanH().to_string()
    assert pointF.aggrFun.to_string() == Sigmoid().to_string()
    assert pointF.lossFun.to_string() == CrossEntropy().to_string()
    assert pointF.learningRate == 10
    assert pointF.momCoeff == 7
    assert pointF.batchSize == 30

def test_mutation():
    hrange = HyperparameterRange((0, 4), (2, 10), [ReLu(), Sigmoid(), Softmax()], [Softmax(), Sigmoid(), TanH(), ReLu()],
                                 [QuadDiff(), CrossEntropy()], (0, 5), (5, 10), (0, 8))
    pointA = AnnPoint(5, 6, 2, 4, ReLu(), Sigmoid(), QuadDiff(), 1, 7, 3)

    ec = EvolvingClassifier()
    ec.hrange = hrange


    random.seed(1001)
    pointB = ec.mutate(pointA, 1, 1, 8*[0])
    pointC = ec.mutate(pointA, 1, 0.25, 8*[0])

    assert pointA.hiddenLayerCount == 2
    assert pointA.neuronCount == 4
    assert pointA.actFun.to_string() == ReLu().to_string()
    assert pointA.aggrFun.to_string() == Sigmoid().to_string()
    assert pointA.lossFun.to_string() == QuadDiff().to_string()
    assert pointA.learningRate == 1
    assert pointA.momCoeff == 7
    assert pointA.batchSize == 3

    # 0, 8.0851, 0, 0, 0, 3.9294, 7.0333, 2.9957
    assert pointB.inputSize == pointA.inputSize
    assert pointB.outputSize == pointA.outputSize
    assert pointB.hiddenLayerCount == 1
    assert pointB.neuronCount == pytest.approx(8.0851, abs=1e-4)
    assert pointB.actFun.to_string() == Sigmoid().to_string()
    assert pointB.aggrFun.to_string() == Softmax().to_string()
    assert pointB.lossFun.to_string() == CrossEntropy().to_string()
    assert pointB.learningRate == pytest.approx(3.9294, abs=1e-4)
    assert pointB.momCoeff == pytest.approx(7.0333, abs=1e-4)
    assert pointB.batchSize == pytest.approx(2.9957, abs=1e-4)

    # 1, 5.1247, 0, 0, 0, 0.0689, 8.1233, 3.9873
    assert pointC.inputSize == pointA.inputSize
    assert pointC.outputSize == pointA.outputSize
    assert pointC.hiddenLayerCount == 1
    assert pointC.neuronCount == pytest.approx(5.1247, abs=1e-4)
    assert pointC.actFun.to_string() == Sigmoid().to_string()
    assert pointC.aggrFun.to_string() == ReLu().to_string()
    assert pointC.lossFun.to_string() == CrossEntropy().to_string()
    assert pointC.learningRate == pytest.approx(0.0689, abs=1e-4)
    assert pointC.momCoeff == pytest.approx(8.1233, abs=1e-4)
    assert pointC.batchSize == pytest.approx(3.9873, abs=1e-4)

    assert ec.mut_performed == 16

def test_select():
    ec = EvolvingClassifier()
    ec.tournament_size = 2

    random.seed(1010)

    list = [[AnnPoint(1, 1, 1, 1, ReLu(), ReLu(), QuadDiff(), 1, 1, 1), 0.6],
            [AnnPoint(2, 2, 2, 2, Sigmoid(), Sigmoid(), QuadDiff(), 2, 2, 2), 0.5],
            [AnnPoint(3, 3, 3, 3, Softmax(), Softmax(), CrossEntropy(), 3, 3, 3), 0.8],
            [AnnPoint(4, 4, 4, 4, TanH(), TanH(), CrossEntropy(), 4, 4, 4), 1.8]]

    selected = ec.select(list)

    assert selected.inputSize == 4
    assert selected.outputSize == 4
    assert selected.hiddenLayerCount == 4
    assert selected.neuronCount == 4
    assert selected.actFun.to_string() == TanH().to_string()
    assert selected.aggrFun.to_string() == TanH().to_string()
    assert selected.lossFun.to_string() == CrossEntropy().to_string()
    assert selected.learningRate == 4
    assert selected.momCoeff == 4
    assert selected.batchSize == 4

random.seed(1010)
print(random.randint(0, 3))
print(random.randint(0, 2))

# # 0, 8.0851, 0, 0, 0, 3.9294, 7.0333, 2.9957
# random.seed(1001)
# print(random.randint(0, 1))
# print(random.uniform(2, 10))
# print(random.randint(0, 1))
# print(random.randint(0, 2))
# print(random.randint(0, 0))
# print(random.uniform(0, 5))
# print(random.uniform(5, 10))
# print(random.uniform(0, 8))
# print("------------------")
#
# # 1, 5.1247, 0, 0, 0, 0.0689, 8.1233, 3.9873
# print(random.randint(0, 1))
# print(random.uniform(2, 6))
# print(random.randint(0, 1))
# print(random.randint(0, 2))
# print(random.randint(0, 0))
# print(random.uniform(0, 2.25))
# print(random.uniform(5.75, 8.25))
# print(random.uniform(1, 5))
#
# test_mutation()



