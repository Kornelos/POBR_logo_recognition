import numpy as np
from numba import njit


@njit
def image_convolution(matrix: np.ndarray, kernel: np.ndarray):
    # assuming kernel is symmetric and odd
    k_size = len(kernel)
    m_height, m_width, channels = matrix.shape

    # iterates through matrix, applies kernel, and sums
    output = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(2, m_height - 2):
        for j in range(2, m_width - 2):
            for ch in range(channels):
                output[i][j][ch] = np.sum(matrix[:, :, ch][i:k_size + i, j:k_size + j] * kernel, dtype=np.uint8)

    return output


@njit
def dilate(m: np.ndarray):
    m_height, m_width, channels = m.shape

    # iterates through matrix, applies kernel, and sums
    output = np.zeros(m.shape, dtype=np.uint8)
    for i in range(1, m_height - 1):
        for j in range(1, m_width - 1):
            if m[i - 1][j - 1][0] == 255 or m[i - 1][j][0] == 255 or \
                    m[i - 1][j + 1][0] == 255 or m[i][j - 1][0] == 255 or \
                    m[i][j + 1][0] == 255 or m[i + 1][j - 1][0] == 255 or \
                    m[i + 1][j][0] == 255 or m[i + 1][j + 1][0] == 255:
                output[i][j] = np.uint(255)
            else:
                output[i][j] = np.uint(0)
    return output


@njit
def erode(m: np.ndarray):
    m_height, m_width, channels = m.shape

    # iterates through matrix, applies kernel, and sums
    output = np.zeros(m.shape, dtype=np.uint8)
    for i in range(1, m_height - 1):
        for j in range(1, m_width - 1):
            if m[i - 1][j - 1][0] == 255 and m[i - 1][j][0] == 255 and \
                    m[i - 1][j + 1][0] == 255 and m[i][j - 1][0] == 255 and \
                    m[i][j + 1][0] == 255 and m[i + 1][j - 1][0] == 255 and m[i + 1][j][0] == 255 and \
                    m[i + 1][j + 1][0] == 255:
                output[i][j] = np.uint(255)
            else:
                output[i][j] = np.uint(0)
    return output


@njit
def apply_threshold(matrix: np.ndarray) -> np.ndarray:
    m_height, m_width, channels = matrix.shape
    output = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(m_height):
        for j in range(m_width):
            output[i][j] = _threshold_in_range(matrix[i][j])
    return output


@njit
def _threshold_in_range(pixel: np.ndarray) -> np.uint8:
    if (160 <= pixel[0] <= 180) and (100 <= pixel[1] <= 255) and (100 <= pixel[2] <= 255) or \
            (0 <= pixel[0] <= 10) and (100 <= pixel[1] <= 255) and (100 <= pixel[2] <= 255):
        return np.uint(255)
    else:
        return np.uint(0)
