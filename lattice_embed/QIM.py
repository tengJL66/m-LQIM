"""
Original Quantization Index Modulation (QIM)
"""
import numpy as np
from lattice_embed.SDCVP import SDCVP
from lattice_embed import Create_Data
import math
from lattice_embed.Lattice.Lattice_basis import Lattice_Basis


def standard_qim(c: np.array, basis: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Move the host signal based on message
    :param c: single host signal
    :param basis: lattice basis
    :param des: messages 秘密消息
    :param a: magnification
    :param representatives: coset representatives
    :return: host signal with message
    """
    coarse_basis = a * basis  # coarse basis
    q = np.dot(coarse_basis, SDCVP(c - np.dot(basis, representatives[des]), coarse_basis, ))  # quantizer
    return q + np.dot(basis, representatives[des])


def standard_qim2(c: np.array, basis: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Move the host signal based on message
    :param c: single host signal
    :param basis: lattice basis
    :param des: messages 秘密消息
    :param a: magnification
    :param representatives: coset representatives
    :return: host signal with message
    """
    coarse_basis = a * basis  # coarse basis
    q = np.dot(coarse_basis, SDCVP(c - np.dot(basis, des), coarse_basis, ))  # quantizer
    return q + np.dot(basis, des)


def cosets(a: int, n: int) -> list[np.array]:
    """
    Calculate the coset representatives
    :param a: magnification
    :param n: dimension of the lattice
    :return: coset representatives
    """
    vector = []
    for i in range(a):
        vector.append(i)
        pass
    representative = [vector] * n
    from itertools import product  # Cartesian product
    representatives = []
    for item in list(product(*representative)):
        representatives.append(np.asarray(item))
    return representatives


def qim_decode(y: np.array, basis: np.array, a: int, representatives: list[np.array]) -> int:
    """
    decode
    :param y: single host signal with message
    :param basis: lattice basis
    :param a: magnification
    :param representatives: coset representatives
    :return: the coset where host signal belongs, i.e. secret message
    """
    x = SDCVP(y, basis, )  # column vector coefficients of host signals
    for i in range(len(representatives)):
        if (x % a == representatives[i]).all():
            break
    return i


if __name__ == '__main__':
    # b = np.array([[1]])
    # b = np.array([[1, 0],
    #               [1, 2]])
    # b = np.array([[2, 1, 1, 1],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    # b = np.array([[2, -1, 0, 0, 0, 0, 0, 0.5],
    #               [0, 1, -1, 0, 0, 0, 0, 0.5],
    #               [0, 0, 1, -1, 0, 0, 0, 0.5],
    #               [0, 0, 0, 1, -1, 0, 0, 0.5],
    #               [0, 0, 0, 0, 1, -1, 0, 0.5],
    #               [0, 0, 0, 0, 0, 1, -1, 0.5],
    #               [0, 0, 0, 0, 0, 0, 1, 0.5],
    #               [0, 0, 0, 0, 0, 0, 0, 0.5]])
    b = Lattice_Basis('Z',5)
    N = b.shape[0]

    alpha = 2  # magnification
    num_carrier = 1280  # number of carriers (or host signals), divisible by dimension

    # coset representatives
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))

    # create host signals 生成
    carrier = Create_Data.carrier(0, 1, num_carrier)  # mean and variance of Gaussian distribution
    # create secret messages 生成
    secret = Create_Data.trans(Create_Data.secret_information(0.7, len(carrier)), length)

    carrier_standard = []  # host signals after standard QIM
    mse_standard = 0  # MSE of single QIM
    count = 0
    for num in range(0, len(carrier), N):
        qim_standard = standard_qim(carrier[num: num + N], b, secret[count], alpha, coset_representatives)
        carrier_standard.append(qim_standard)
        mse_standard += sum(pow(qim_standard - carrier[num: num + N], 2)) / N

        count += 1
        pass

    J = alpha * np.eye(N)
    R = 1 / N * math.log(np.linalg.det(J), 2)
    print('code rate')
    print(R)
    MSE_standard = mse_standard / len(carrier)
    print('MSE of QIM')
    print(MSE_standard)
    PSNR_standard = 20 * math.log(255 / math.sqrt(MSE_standard), 10)
    print('PSNR of QIM')
    print(PSNR_standard)


    # decode
    m_standard_decode = []  # messages after decoding
    for num in range(0, round(len(carrier) / N)):
        m_standard_decode.append(qim_decode(carrier_standard[num], b, alpha, coset_representatives))
        pass
    difference_standard = 0
    for num in range(len(m_standard_decode)):
        if m_standard_decode[num] != secret[num]:
            difference_standard += 1
            pass
        pass
    print('difference after decoding')
    print(difference_standard)


