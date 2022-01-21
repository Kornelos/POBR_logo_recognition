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
def apply_threshold(matrix: np.ndarray, hue_value: np.uint8) -> np.ndarray:
    m_height, m_width, channels = matrix.shape
    output = np.zeros(matrix.shape, dtype=np.uint8)  # todo: change to 1D
    for i in range(m_height):
        for j in range(m_width):
            output[i][j] = _threshold_in_range(matrix[i][j])
    return output


@njit
def _threshold_in_range(pixel: np.ndarray) -> np.uint8:
    # lower_red = np.array([160, 50, 50])
    # upper_red = np.array([180, 255, 255])
    # lower_red2 = np.array([0, 70, 50])
    # upper_red2 = np.array([10, 255, 255])
    if (160 <= pixel[0] <= 180) and (50 <= pixel[1] <= 255) and (50 <= pixel[2] <= 255) or \
            (0 <= pixel[0] <= 10) and (70 <= pixel[1] <= 255) and (70 <= pixel[2] <= 255):
        return np.uint(255)
    else:
        return np.uint(0)