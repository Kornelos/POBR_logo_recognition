from queue import Queue
from typing import Tuple

import cv2
import numpy as np
from numba import njit
from numba.experimental import jitclass


class Detected:
    selected = np.array(1)
    pixelCount = 0
    bounding_box = (0, 0, 0, 0)

    def __init__(self, matrix: np.ndarray, used: np.ndarray, idx: Tuple[int, int]):
        q = Queue(-1)
        q.put(idx)
        segment = np.zeros(used.shape, dtype=bool)
        m_height, m_width, _ = matrix.shape
        pixel_count = 0
        while not q.empty():
            i, j = q.get()
            if not used[i][j]:
                used[i][j] = True
                if not np.array_equal(matrix[i][j], np.array([0, 0, 0])):
                    pixel_count += 1
                    segment[i][j] = True

                    # check neighbours
                    if j - 1 >= 0:
                        q.put((i, j - 1))
                    if i - 1 >= 0:
                        q.put((i - 1, j))
                    if j + 1 < m_width:
                        q.put((i, j + 1))
                    if i + 1 < m_height:
                        q.put((i + 1, j))
        self.selected = segment
        self.pixel_count = pixel_count
        self.bounding_box = self._create_bbox()

    def _create_bbox(self) -> Tuple[int, int, int, int]:
        m_height, m_width = self.selected.shape
        i_min = m_height
        j_min = m_width
        i_max = 0
        j_max = 0
        for i in range(m_height):
            for j in range(m_width):
                if self.selected[i][j]:
                    if i < i_min:
                        i_min = i
                    if j < j_min:
                        j_min = j
                    if i > i_max:
                        i_max = i
                    if j > j_max:
                        j_max = j
        return i_max, j_max, i_min, j_min
