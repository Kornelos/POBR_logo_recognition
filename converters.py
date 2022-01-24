import numpy as np
from numba import njit


@njit
def bgr2hsv(matrix: np.ndarray) -> np.ndarray:
    m_height, m_width, channels = matrix.shape
    output = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(m_height):
        for j in range(m_width):
            output[i][j] = _bgr2hsv_pixel_fast(matrix[i][j])
    return output


@njit
def _bgr2hsv_pixel_fast(bgr_pixel: np.ndarray) -> np.ndarray:
    # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    hsv = np.zeros(3, dtype=np.uint8)
    bgr_min = np.min(bgr_pixel)
    bgr_max = np.max(bgr_pixel)
    hsv[2] = bgr_max
    if hsv[2] == 0:
        return hsv
    hsv[1] = 255 * (bgr_max - bgr_min) / hsv[2]
    if hsv[1] == 0:
        return hsv
    delta = bgr_max - bgr_min
    if bgr_max == bgr_pixel[2]:
        hsv[0] = 0 + 43 * (np.int16(bgr_pixel[1]) - np.int16(bgr_pixel[0])) / delta
    elif bgr_max == bgr_pixel[1]:
        hsv[0] = 85 + 43 * (np.int16(bgr_pixel[0]) - np.int16(bgr_pixel[2])) / delta
    else:
        hsv[0] = 171 + 43 * (np.int16(bgr_pixel[2]) - np.int16(bgr_pixel[1])) / delta
    return hsv
