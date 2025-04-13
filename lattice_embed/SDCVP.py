"""
Find the nearest lattice vector of a given vector under lattice basis of any dimension
"""
import copy
import numpy as np
import math


def SDCVP(y: np.array, H: np.array, radius=1e100) -> np.array:
    """
    Compute the nearest vector to a given arbitrary vector in the lattice,
    using the spherical decoding algorithm to calculate
    :param y: A given vector in n-dimensional Euclidean space
    :param H: Lattice basis
    :param radius: Initial radius for spherical decoding
    :return: Lattice basis's integer coefficients of the nearest lattice vector
    """
    if H.shape[0] < H.shape[1]:
        H = np.vstack((H, np.zeros((H.shape[1] - H.shape[0], H.shape[1]))))
    n = H.shape[1]
    Q, R = qr(H)
    y2 = y.reshape((-1, 1))
    if y.size != H.shape[0]:
        return "dimension error"
    z = np.dot(Q.T, y2)

    global SPHDEC_RADIUS
    global RETVAL
    global x
    global NUMX
    SPHDEC_RADIUS = radius
    RETVAL = np.zeros((n, 1))
    x = np.zeros((n, 1))
    NUMX = 0

    sphdec_core(z, R, n, 0)

    if NUMX > 0:
        r = RETVAL[:]
    else:
        r = np.zeros((n, 1))
    return r.reshape((-1))


def sphdec_core(z: np.array, R: np.array, layer: int, dist: int):
    """
    Sphere Decoding Algorithm
    :param z: ???
    :param R: Upper triangular matrix
    :param layer: Recursion level
    :param dist: distance of ???
    :return:
    """
    global SPHDEC_RADIUS
    global RETVAL
    global x
    global NUMX

    n = R.shape[1]
    if layer == n:
        zi = z
        pass
    else:
        zi = z - np.dot(R[:, layer:], x[layer:])
        pass

    row = (layer - 1) % zi.shape[0]
    column = math.floor((layer - 1) / zi.shape[0])
    # zi[layer-1] == zi[(layer - 1) % zi.shape[0], int((layer - 1) / zi.shape[0])]
    c = TheRound(zi[row, column] / R[layer - 1, layer - 1])
    x[row, column] = c
    d = pow(z[row, column] - np.dot(R[layer - 1, layer - 1:], x[layer - 1:]), 2) + dist
    # d = round(d[0], 4)
    if d <= SPHDEC_RADIUS:
        if layer == 1:
            RETVAL = copy.deepcopy(x)
            SPHDEC_RADIUS = d
            NUMX = NUMX + 1
            pass
        else:
            sphdec_core(z, R, layer - 1, d)
            pass
        pass

    delta = 0
    while d <= SPHDEC_RADIUS:
        delta = delta + 1
        for k in [1, 2]:
            ci = c + delta * pow(-1, k)
            x[row, column] = ci
            d = pow(z[row, column] - np.dot(R[layer - 1, layer - 1:], x[layer - 1:]), 2) + dist
            # d = round(d[0], 4)
            if d <= SPHDEC_RADIUS:
                if layer == 1:
                    RETVAL = copy.deepcopy(x)
                    SPHDEC_RADIUS = d
                    NUMX = NUMX + 1
                    pass
                else:
                    sphdec_core(z, R, layer - 1, d)
                    pass
                pass
            pass
        pass
    pass


def qr(mat: np.array):
    """
    QR decomposition
    :param mat: A given matrix, here is the lattice basis
    :return: Orthogonal matrix Q and upper triangular matrix R
    """
    row_o, col_o = mat.shape
    if row_o > col_o:
        mat = np.hstack((mat, np.zeros((row_o, row_o - col_o))))
        pass
    rows, cols = mat.shape
    R = np.copy(mat)
    Q = np.eye(cols)
    for col in range(cols):
        for row in range(col + 1, rows):
            if abs(R[row, col]) < 1e-6:
                continue
            f = R[col, col]
            s = R[row, col]
            den = np.sqrt(f * f + s * s)
            c = f / den
            s = s / den

            T = np.eye(rows)
            T[col, col], T[row, row] = c, c
            T[row, col], T[col, row] = -s, s

            R = T.dot(R)
            Q = T.dot(Q)
    Q = Q.T[:, :col_o]
    R = R[:col_o, :col_o]
    return np.round(Q, 15), np.round(R, 15)
    # return Q, R


def TheRound(x: float):
    """
    Deal with the problem existing in python:
    if the value of round() is as far away from the integers on both sides, it will be kept on the even side.
    For example: round(1.5) == round(2.5) == 2 in python, which is wrong.
    :param x: A given number
    :return: Rounded number
    """
    flag = True
    if x < 0:
        flag = False
    y = abs(x)
    if y % 1 == 0.5:
        y = math.ceil(y)
    else:
        y = round(y)
    if x < 0:
        return 0 - y
    else:
        return y


if __name__ == '__main__':
    basis = np
    # basis = np.array([[1, 1 / 2],
    #                   [0, pow(3, 0.5) / 2]])
    basis = np.array([[2, 1, 1, 1],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    # basis = np.array([[2, -1, 0, 0, 0, 0, 0, 0.5],
    #                   [0, 1, -1, 0, 0, 0, 0, 0.5],
    #                   [0, 0, 1, -1, 0, 0, 0, 0.5],
    #                   [0, 0, 0, 1, -1, 0, 0, 0.5],
    #                   [0, 0, 0, 0, 1, -1, 0, 0.5],
    #                   [0, 0, 0, 0, 0, 1, -1, 0.5],
    #                   [0, 0, 0, 0, 0, 0, 1, 0.5],
    #                   [0, 0, 0, 0, 0, 0, 0, 0.5]])
    vector = np.array([1, 2, 5, 1])
    ans = SDCVP(vector, basis, )
    print(ans)
