import numpy as np

def get_mask(input_size: int, output_size: int, neuron_count: int) -> np.ndarray:
    mask = np.zeros((neuron_count, neuron_count))
    mask[:-output_size, input_size:] = 1
    mask[:input_size, -output_size:] = 0
    # mask = np.triu(mask)
    np.fill_diagonal(mask, 0)

    return mask