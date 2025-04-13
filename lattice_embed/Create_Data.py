import numpy as np
import random


def carrier(mu, sigma, num) -> np.array:
    """
    Integer Gaussian distribution
    :param mu: mean
    :param sigma: variance
    :param num: number of carrier
    :return: Host signal
    """
    c = []
    i = 0
    while i < num:
        cc = round(random.normalvariate(mu, sigma))
        if -256 <= cc <= 256:
            c.append(cc)
            i += 1
            pass
        pass
    return np.asarray(c)


def secret_information(rate, num) -> list:
    """
    Generate secret message bits in certain probability
    :param rate: probability of 0
    :param num: number of secret
    :return: secret message bits
    """
    secret_list = []
    for i in range(num):
        r = random.random()
        if r < rate:
            secret_list.append(0)
        else:
            secret_list.append(1)
        pass
    return secret_list


# 将秘密消息比特序列转换为存储所需格式
def trans(secret_bits: list[int], dimension) -> list:
    """
    Convert the secret message bit sequence into the format required for storage
    :param secret_bits: secret message bit sequence
    :param dimension: dimension of lattice
    :return: converted message sequence
    """
    trans_list = []
    for i in range(0, len(secret_bits), dimension):
        t = ""
        for j in range(i, i+dimension):
            t += str(secret_bits[j])
        trans_list.append(int("0b" + t, 2))
    return trans_list


def trans_back(encrypt):
    return bin(encrypt).replace("0b", "")


if __name__ == '__main__':
    N = 2
    num_carrier = 100
    m = carrier(0, 1, num_carrier)
    print(m)

    secret = secret_information(0.8, N*num_carrier)
    print(secret)
    print(trans(secret, N))

