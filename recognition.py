from math import pi, sqrt
from typing import Tuple

import numpy as np
from numba import njit

from configuration import W3_LETTER_MAX, W3_LETTER_MIN, M1_LETTER_MIN, M1_LETTER_MAX, M7_LETTER_MIN, M7_LETTER_MAX, \
    W3_LOGO_MAX, W3_LOGO_MIN, M1_LOGO_MAX, M1_LOGO_MIN, M7_LOGO_MAX, M7_LOGO_MIN


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
    # print(f"w3: {w3}  M1: {M1} M7: {M7}")
    if W3_LETTER_MAX > w3 > W3_LETTER_MIN and M1_LETTER_MAX > M1 > M1_LETTER_MIN and M7_LETTER_MAX > M7 > M7_LETTER_MIN:  # merged HM
        print(f"found hm: w3: {w3}  M1: {M1} M7: {M7}")
        return True, False
    elif W3_LOGO_MAX > w3 > W3_LOGO_MIN and M1_LOGO_MAX > M1 > M1_LOGO_MIN and M7_LOGO_MAX > M7 > M7_LOGO_MIN:  # separate H M
        print(f"found h or m: w3: {w3}  M1: {M1} M7: {M7}")
        return True, True
    else:
        return False, False
