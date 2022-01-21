from typing import Tuple, List

import numpy as np
from numba import njit

PIXEL_COUNT_MIN = 1000


@njit
def flood_fill(matrix: np.ndarray) -> List[Tuple[
                np.ndarray, int, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """:returns List[Tuple[segment, pixel_count, bounding_box(segment)]]"""

    m_height, m_width, _ = matrix.shape
    # output = np.zeros(matrix.shape, dtype=np.uint8)
    used = np.zeros((m_height, m_width), dtype=np.uint8)

    detected = []
    for i in range(m_height):
        for j in range(m_width):
            if not used[i][j]:
                if not np.array_equal(matrix[i][j], np.array([0, 0, 0])):
                    # new element detected
                    d = detect_object(matrix, used, (i, j))
                    if d[1] > PIXEL_COUNT_MIN:
                        detected.append(d)
    return detected


@njit
def detect_object(matrix: np.ndarray, used: np.ndarray, idx: Tuple[int, int]) -> Tuple[
                    np.ndarray, int, Tuple[Tuple[int, int], Tuple[int, int]]]:
    q = []
    q.append(idx)
    segment = np.zeros(used.shape, dtype=np.uint8)
    m_height, m_width, _ = matrix.shape
    pixel_count = 0
    while q:
        i, j = q.pop()
        if not used[i][j]:
            used[i][j] = True
            if not np.array_equal(matrix[i][j], np.array([0, 0, 0])):
                pixel_count += 1
                segment[i][j] = True

                # check neighbours
                if j - 1 >= 0:
                    q.append((i, j - 1))
                if i - 1 >= 0:
                    q.append((i - 1, j))
                if j + 1 < m_width:
                    q.append((i, j + 1))
                if i + 1 < m_height:
                    q.append((i + 1, j))
    return segment, pixel_count, _create_bbox(segment)


@njit
def _create_bbox(segment) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    m_height, m_width = segment.shape
    i_min = m_height
    j_min = m_width
    i_max = 0
    j_max = 0
    for i in range(m_height):
        for j in range(m_width):
            if segment[i][j]:
                if i < i_min:
                    i_min = i
                if j < j_min:
                    j_min = j
                if i > i_max:
                    i_max = i
                if j > j_max:
                    j_max = j
    return (j_min, i_min), (j_max, i_max)
