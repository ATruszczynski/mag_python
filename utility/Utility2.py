from statistics import mean

import numpy as np

from ann_point.Functions import ActFun


def get_weight_mask(input_size: int, output_size: int, neuron_count: int) -> np.ndarray:
    mask = np.zeros((neuron_count, neuron_count))
    mask[:-output_size, input_size:] = 1
    mask[:input_size, -output_size:] = 0
    np.fill_diagonal(mask, 0)

    return mask

def get_bias_mask(input_size: int, neuron_count: int) -> np.ndarray:
    bias_mask = np.ones((1, neuron_count))
    bias_mask[0, :input_size] = 0

    return bias_mask

def accuracy(confusion_matrix: np.ndarray):
    tot_sum = np.sum(confusion_matrix)
    diag_sum = np.sum(np.diag(confusion_matrix))

    accuracy = diag_sum / tot_sum

    return accuracy

def average_precision(conf_matrix):
    return np.average(get_precisions(conf_matrix))

def get_precisions(conf_matrix) -> [float]:
    col_sum = np.sum(conf_matrix, axis=0)
    row_sum = np.sum(conf_matrix, axis=1)
    diag = np.diag(conf_matrix)

    class_prec = []

    for i in range(len(col_sum)):
        if row_sum[i] > 0:
            if col_sum[i] > 0:
                class_prec.append(diag[i] / col_sum[i])
            else:
                class_prec.append(0)

    return class_prec


def average_recall(conf_matrix):
    return np.average(get_recalls(conf_matrix))

def get_recalls(conf_matrix) -> [float]:
    row_sums = np.sum(conf_matrix, axis=1)
    diag = np.diag(conf_matrix)

    class_recalls = []

    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            class_recalls.append(diag[i] / row_sums[i])

    return class_recalls

def efficiency(conf_matrix):
    acc = accuracy(conf_matrix)
    prec = average_precision(conf_matrix)
    rec = average_recall(conf_matrix)
    f1 = average_f1_score(conf_matrix)

    return mean([acc, prec, rec, f1])

def m_efficiency(conf_matrix):
    acc = accuracy(conf_matrix)
    prec = np.min(get_precisions(conf_matrix))
    rec = np.min(get_recalls(conf_matrix))
    f1 = np.min(get_f1_scores(conf_matrix))

    return min([acc, prec, rec, f1])

def get_f1_scores(conf_matrix):
    precisions = get_precisions(conf_matrix)
    recall = get_recalls(conf_matrix)

    f1 = []
    for i in range(len(precisions)):
        prec = precisions[i]
        rec = recall[i]
        numerator = 2 * prec * rec
        denominator = prec + rec

        if denominator == 0:
            f1.append(0)
        else:
            f1.append(numerator / denominator)

    return f1

def average_f1_score(conf_matrix):
    f1 = get_f1_scores(conf_matrix)

    return mean(f1)

def check_cond_in_cn_const(cond: bool):
    if not cond:
        raise CNConstructorException()


class CNConstructorException(Exception):
    def __init__(self):
        pass


def assert_acts_same(acts1: [ActFun], acts2: [ActFun]):
    assert len(acts1) == len(acts2)

    for i in range(len(acts1)):
        assert acts1[i].to_string() == acts2[i].to_string()

