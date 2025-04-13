import numpy as np
from Lattice_packing import Lattice_packing
from Lattice_basis import Lattice_Basis
from Lattice_covering import Lattice_covering
from Lattice_mean_square import Lattice_Mean_Square


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
