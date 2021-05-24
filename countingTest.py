from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from evolving_classifier.EvolvingClassifier import EvolvingClassifier
from utility.Utility import *
from neural_network.FeedForwardNeuralNetwork import *
from statistics import mean

if __name__ == '__main__':
    random.seed(10011111)

    # sm = Softmax()
    # print(sm.prec_der(np.array([[1], [2], [3]])))
    # print(sm.computeDer(np.array([[1], [2], [3]])))

    count = 500
    size = 5
    x,y = generate_counting_problem(count, size)
    X,Y = generate_counting_problem(ceil(count/5), size)

    ec = EvolvingClassifier()
    ec.hrange.neuronCount = 0
    ec.prepare(200, 200, 0.8, 0.05, 2, (x, y, X, Y), 1001)
    npoint = ec.run(30, 12)
    print(npoint.links)
    print(npoint.to_string())
    tests = npoint.test(X, Y)
    print(tests[:3])
    print(tests[3])
    print(mean(tests[:3]))

    ori = 1

