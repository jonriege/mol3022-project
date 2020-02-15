import random as rnd
import numpy as np

KINGDOMS = ['ARCHAEA', 'POSITIVE', 'EUKARYA', 'NEGATIVE']
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYX'


def read_file(filename):
    with open(filename, 'r') as handle:
        data = [line.rstrip() for line in handle]
    return data


def one_hot_encode_kingdom(kingdom):
    x = np.zeros(len(KINGDOMS))
    idx = KINGDOMS.index(kingdom)
    x[idx] = 1
    return x


def one_hot_encode_amino_acid(amino_acid):
    x = np.zeros(len(AMINO_ACIDS))
    idx = AMINO_ACIDS.index(amino_acid)
    x[idx] = 1
    return x


def process_datapoint(x1, x2):
    x = []
    _, kingdom, sp, _ = x1.split('|')

    if sp == 'NO_SP':
        y = 0
    else:
        y = 1

    x.extend(one_hot_encode_kingdom(kingdom))

    x2 += (70 - len(x2)) * 'X'
    for amino_acid in x2:
        x.extend(one_hot_encode_amino_acid(amino_acid))

    x = np.array(x)
    return x, y


def read_data(filename):
    p_data = []
    data = read_file(filename)
    for x1, x2, _ in zip(*[iter(data)] * 3):
        x, y = process_datapoint(x1, x2)
        p_data.append((x, y))
    rnd.shuffle(p_data)
    p_data = p_data[:3000]
    X = np.array([d[0] for d in p_data])
    y = np.array([d[1] for d in p_data])
    return X, y
