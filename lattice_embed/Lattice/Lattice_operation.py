import numpy as np


def lattice_information(lattice_name: str, N: int = 2):
    type = 1
    if lattice_name == 'Z':
        B = Lattice_Basis(lattice_name, N)
        rp = Lattice_packing(B)
        B = 0.5 * B / rp * type
        rp = 0.5 * rp / rp * type
        rc = pow(N, 1 / 2) * rp * type
        pass
    else:
        B = Lattice_Basis(lattice_name, N)
        rp = Lattice_packing(B)
        B = 0.5 * B / rp * type
        rp = 0.5 * rp / rp * type
        rc = Lattice_covering(lattice_name) * rp * type
        pass
    G = Lattice_Mean_Square(lattice_name)
    return B, rp, rc, G


def Lattice_Basis(a: str, N: int = 0) -> np.array:
    if a == 'Z':
        if N <= 0:
            print('please input dimension of Z lattice')
        else:
            m = np.identity(N)
        pass
    elif a == 'A2':
        m = np.array([[pow(3, 1 / 2) / 2, 0],
                      [0.5, 1]])
        pass
    elif a == 'A3-self':
        m = np.array([[1.0000, 0.5000, 0.5000],
                      [0, 0.8660, 0.2887],
                      [0, 0, 0.8165]])
        pass
    elif a == 'A3':
        m = np.array([[-1, 1, 0],
                      [-1, -1, 1],
                      [0, 0, -1]])
        pass
    elif a == 'A3*':
        m = np.array([[2, 0, 1],
                      [0, 2, 1],
                      [0, 0, 1]])
        pass
    elif a == 'D4':
        m = np.array([[2, 1, 1, 1],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        pass
    elif a == 'D5':
        m = np.array([[1, 0, 0, 0, 0.5],
                      [0, 1, 0, 0, 0.5],
                      [0, 0, 1, 0, 0.5],
                      [0, 0, 0, 1, 0.5],
                      [0, 0, 0, 0, 0.5]])
        pass
    elif a == 'E6':
        m = np.array([[0, 0, 0, 0, 0, 0.5],
                      [-1, 0, 0, 0, 0, 0.5],
                      [1, -1, 0, 0, 0, 0.5],
                      [0, 1, -1, 0, 0, 0.5],
                      [0, 0, 1, -1, 0, 0.5],
                      [0, 0, 0, 1, -1, 0.5],
                      [0, 0, 0, 0, 1, 0.5],
                      [0, 0, 0, 0, 0, 0.5]])
        pass
    elif a == 'E7':
        m = np.array([[-1, 0, 0, 0, 0, 0, 0.5],
                      [1, -1, 0, 0, 0, 0, 0.5],
                      [0, 1, -1, 0, 0, 0, 0.5],
                      [0, 0, 1, -1, 0, 0, 0.5],
                      [0, 0, 0, 1, -1, 0, 0.5],
                      [0, 0, 0, 0, 1, -1, 0.5],
                      [0, 0, 0, 0, 0, 1, 0.5],
                      [0, 0, 0, 0, 0, 0, 0.5]])
        pass
    elif a == 'E8':
        m = np.array([[2, -1, 0, 0, 0, 0, 0, 0.5],
                      [0, 1, -1, 0, 0, 0, 0, 0.5],
                      [0, 0, 1, -1, 0, 0, 0, 0.5],
                      [0, 0, 0, 1, -1, 0, 0, 0.5],
                      [0, 0, 0, 0, 1, -1, 0, 0.5],
                      [0, 0, 0, 0, 0, 1, -1, 0.5],
                      [0, 0, 0, 0, 0, 0, 1, 0.5],
                      [0, 0, 0, 0, 0, 0, 0, 0.5]])
        pass
    return m


def Lattice_packing(a: np.array) -> float:
    p = 2
    return min(pow(sum(pow(a, p)), 1 / p)) / 2


def Lattice_covering(a: str) -> float:
    if a == '1':
        m = 1
    elif a == 'Z' or a == 'A3' or a == 'D4' or a == 'E8':
        m = pow(2, 1 / 2)
    elif a == 'A2':
        m = 2 / pow(3, 1 / 2)
    return m


def Lattice_Mean_Square(a: str) -> float:
    if a == '1' or a == 'Z':
        g = 1 / 12
    elif a == 'A2':
        g = 5 / 36 / pow(3, 1 / 2)
    elif a == 'A3':
        g = 0.078543
    elif a == 'D4':
        g = 0.076603
    elif a == 'E8':
        g = 0.071682
    return g
