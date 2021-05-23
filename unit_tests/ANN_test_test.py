from neural_network.FeedForwardNeuralNetwork import *

def test_nn_test():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]])]
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=1, neuronCount=2, actFun=TanH(),
                                       aggrFun=Softmax(), lossFun=CrossEntropy(), learningRate=-3, momCoeffL=-3, batchSize=-2, seed=1016)



network = FeedForwardNeuralNetwork(inputSize=2, outputSize=3, hiddenLayerCount=1, neuronCount=2, actFun=TanH(),
                                   aggrFun=Softmax(), lossFun=CrossEntropy(), learningRate=-3, momCoeffL=-3, batchSize=-2, seed=1016)