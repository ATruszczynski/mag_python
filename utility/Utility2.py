import numpy as np

#TODO TESST!!!!!!!!!!!!!!
def get_weight_mask(input_size: int, output_size: int, neuron_count: int) -> np.ndarray:
    mask = np.zeros((neuron_count, neuron_count))
    mask[:-output_size, input_size:] = 1
    mask[:input_size, -output_size:] = 0
    # mask = np.triu(mask)
    np.fill_diagonal(mask, 0)

    return mask

def get_bias_mask(input_size: int, neuron_count: int) -> np.ndarray:
    bias_mask = np.ones((1, neuron_count))
    bias_mask[0, :input_size] = 0

    return bias_mask