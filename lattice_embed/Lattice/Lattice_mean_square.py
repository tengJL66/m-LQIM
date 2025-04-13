def Lattice_Mean_Square(a: str) -> float:
    if a == '1' or a == 'Z':
        g = 1 / 12
    elif a == 'A2':
        g = 5 / 36 / pow(3, 1/2)
    elif a == 'A3' :
        g = 0.078543
    elif a == 'D4' :
        g = 0.076603
    elif a == 'E8':
        g = 0.071682
    return g
