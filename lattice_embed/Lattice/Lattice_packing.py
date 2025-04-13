import numpy as np


def Lattice_packing(a: np.array) -> float:
    p = 2
    return min(pow(sum(pow(a, p)), 1 / p)) / 2
