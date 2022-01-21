from math import pi, sqrt

# w3
# """Współczynnik kształtu Malinowskiej"""
from typing import Tuple

import numpy as np
from numba import njit


@njit
def calculate_w3(circumference: int, area: int) -> float:
    return circumference / (2 * sqrt(pi * area)) - 1


@njit
def is_on_circumference(matrix: np.ndarray, idx: Tuple[int, int]) -> bool:
    i, j = idx
    if matrix[i][j] == 1 and has_neighbour_outside(i, j, matrix):
        return True
    else:
        return False


@njit
def has_neighbour_outside(i: int, j: int, matrix: np.ndarray) -> bool:
    return (matrix[i - 1][j] == 0 or matrix[i][j - 1] == 0 or matrix[i + 1][j] == 0 or matrix[i][j + 1] == 0
            or matrix[i + 1][j - 1] == 0 or matrix[i + 1][j - 1] == 0 or matrix[i - 1][j - 1] == 0 or matrix[i - 1][
                j + 1] == 0)


@njit
def calculate_circumference(matrix: np.ndarray) -> int:
    m_height, m_width = matrix.shape
    circ = 0
    for i in range(1, m_height - 1):
        for j in range(1, m_width - 1):
            if is_on_circumference(matrix, (i, j)):
                circ += 1
    return circ


# moments (we use M1 and M7)
# m_00 = area
# M1 = M_20 + M_02 / m ^ 2_00
# M_20 = m_20 - m ^ 2_10 / m_00
# M_02 = m_02 - m ^ 2_01 / m_00


# function to calculate m with index xy
@njit
def calculate_m_xy(x: int, y: int, matrix: np.ndarray) -> float:
    sum = 0.0
    m_height, m_width = matrix.shape
    for i in range(m_height):
        for j in range(m_width):
            if matrix[i][j] == 0:
                sum += pow(i, x) * pow(j, y)
    return sum


@njit
def basic_moments(matrix):
    m00 = calculate_m_xy(0, 0, matrix)
    M_20 = calculate_m_xy(2, 0, matrix) - (pow(calculate_m_xy(1, 0, matrix), 2) / m00)
    M_02 = calculate_m_xy(0, 2, matrix) - (pow(calculate_m_xy(0, 1, matrix), 2) / m00)
    return M_02, M_20, m00


@njit
def calculate_M1(matrix: np.ndarray) -> float:
    M_02, M_20, m00 = basic_moments(matrix)
    return (M_20 + M_02) / pow(m00, 2)


@njit
def calculate_M7(matrix: np.ndarray) -> float:
    M_02, M_20, m00 = basic_moments(matrix)
    M_11 = calculate_m_xy(1, 1, matrix) - ((calculate_m_xy(1, 0, matrix) * calculate_m_xy(0, 1, matrix)) / m00)
    return (M_20 * M_02 - pow(M_11, 2)) / pow(m00, 4)


def is_hm(matrix: np.ndarray, pixel_count: int) -> Tuple[bool, bool]:
    """:returns (is_logo, should_merge) should be merged if its h or m"""
    circ = calculate_circumference(matrix)
    area = pixel_count
    w3 = calculate_w3(circ, area)
    M1 = calculate_M1(matrix)
    M7 = calculate_M7(matrix)
    print(f"w3: {w3}  M1: {M1} M7: {M7}")
    if 3.5 > w3 > 2 and 0.18 > M1 > 0.17 and 0.008 > M7 > 0.007 or \
        2 > w3 > 1.9 and 0.18 > M1 > 0.17 and 0.008 > M7 > 0.007 or \
        (2.2 > w3 > 1.7 and 0.17 > M1 > 0.16 and 0.007 > M7 > 0.0068):  # separate H M
        print(f"found h or m: w3: {w3}  M1: {M1} M7: {M7}")
        return True, True
    elif (3.6 > w3 > 3.4 and 0.17 > M1 > 0.16 and 0.0075 > M7 > 0.007) or \
            (2 > w3 > 1.9 and 0.17 > M1 > 0.16 and 0.007 > M7 > 0.0068):  # merged HM
        print(f"found hm: w3: {w3}  M1: {M1} M7: {M7}")
        return True, False
    else:
        return False, False
# example img
# H: w3: 2.476006191789203  M1: 0.17507523069129763 M7: 0.007662793675624868
# M: w3: 3.271185161794998  M1: 0.1788113658425772 M7: 0.00799330237645096
# HM merged: w3: 3.502647829163169  M1: 0.16781878475273718 M7: 0.0070405422616266556

# img 5
# H: w3: 1.779726856990628  M1: 0.1670848983184895 M7: 0.006979300906993466
# M: w3: 2.1094828052762757  M1: 0.1670631739173778 M7: 0.006977411064883311
