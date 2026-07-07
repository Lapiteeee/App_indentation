import numpy as np


def second_derivative(f_2_0, f_1_0, f_0_0, f_neg1_0, f_neg2_0, h):
    return (1.0 / (12.0 * h ** 2)) * (-f_2_0 + 16.0 * f_1_0 - 30.0 * f_0_0 + 16.0 * f_neg1_0 - f_neg2_0)


def calculate_K(x, y, E=210000.0):
    return (1.0 - y) * np.log(x / E)


def calculate_sigma_y(K, y, E=210000.0):
    return E * np.exp(K / (1.0 - y))


def loi_hollomon(sigma_y, n, deformation, E=210000.0):
    if deformation <= 0:
        return 0.0
    return (sigma_y ** (1 - n)) * (E ** n) * (deformation ** n)


def calculer_integrale(x, y1, y2):
    if len(x) != len(y1) or len(x) != len(y2):
        raise ValueError("Les tableaux x, y1 et y2 doivent avoir la meme taille.")
    difference = y1 - y2
    carre_difference = difference ** 2
    dx = np.diff(x)
    return np.sum(carre_difference[:-1] * dx)
