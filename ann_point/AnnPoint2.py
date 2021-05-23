from math import ceil

from ann_point.Functions import *

class AnnPoint2():
    def __init__(self, input_size: int, output_size: int, hiddenNeuronCounts: [int], activationFuns: [ActFun], weights: [np.ndarray], biases: [np.ndarray]):
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_neuron_counts = []
        for i in range(len(hiddenNeuronCounts)):
            self.hidden_neuron_counts.append(hiddenNeuronCounts[i])

        self.activation_functions = []
        for i in range(len(activationFuns)):
            self.activation_functions.append(activationFuns[i].copy())

        self.weights = []
        for i in range(len(weights)):
            self.weights.append(weights[i].copy()) # TODO test if deep copy

        self.biases = []
        for i in range(len(biases)):
            self.biases.append(biases[i].copy()) # TODO test if deep copy

    def get_full_neuron_counts(self) -> [int]:
        neuron_counts = [self.input_size]
        for i in range(len(self.hidden_neuron_counts)):
            neuron_counts.append(self.hidden_neuron_counts[i])
        neuron_counts.append(self.output_size)
        return neuron_counts

    def into_numbered_layer_tuples(self) -> [[int, int, ActFun, np.ndarray, np.ndarray]]:
        result = [[0, self.input_size, None, None, None]]
        neuron_counts = self.get_full_neuron_counts()
        for i in range(1, len(neuron_counts)):
            result.append([i, neuron_counts[i], self.activation_functions[i - 1].copy(), self.weights[i - 1].copy(), self.biases[i - 1].copy()])
        return result


    def copy(self):
        result = AnnPoint2(input_size=self.input_size, output_size=self.output_size, hiddenNeuronCounts=self.hidden_neuron_counts, activationFuns=self.activation_functions, weights=self.weights, biases=self.biases)

        return result

    def to_string(self):
        result = f"|{self.input_size}|"

        for i in range(len(self.hidden_neuron_counts)):
            result += f"{self.activation_functions[i].to_string()}|{self.hidden_neuron_counts[i]}|"

        result += f"{self.output_size}|"

        return result

    def to_string_full(self): # TODO test
        result = ""
        result += "placeholder"

        return result

    def size(self):
        result = 0

        for i in range(len(self.hidden_neuron_counts) - 1):
            result += self.hidden_neuron_counts[i] * self.hidden_neuron_counts[i + 1]

        # TODO fix

        return result
    #TODO kiedyś się pojawił problem tego, że sieć mogła przetrwać zmianę input_size
def point_from_layer_tuples(layer_tuples: [int, int, ActFun, np.ndarray, np.ndarray]) -> AnnPoint2:
    input_size = layer_tuples[0][1]
    output_size = layer_tuples[-1][1]

    hidden_neur = []
    actFuns = []
    weights = []
    biasess = []

    for i in range(1, len(layer_tuples)):
        layer_tuple = layer_tuples[i]

        if i != len(layer_tuples) - 1:
            hidden_neur.append(layer_tuple[1])
        actFuns.append(layer_tuple[2])
        weights.append(layer_tuple[3])
        biasess.append(layer_tuple[4])

    return AnnPoint2(input_size=input_size, output_size=output_size, hiddenNeuronCounts=hidden_neur, activationFuns=actFuns, weights=weights, biases=biasess)

