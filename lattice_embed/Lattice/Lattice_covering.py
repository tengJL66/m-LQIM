def Lattice_covering(a: str) -> float:
    if a == '1':
        m = 1
    elif a == 'Z' or a == 'A3' or a == 'D4' or a == 'E8':
        m = pow(2, 1/2)
    elif a == 'A2':
        m = 2 / pow(3, 1/2)
    return m
