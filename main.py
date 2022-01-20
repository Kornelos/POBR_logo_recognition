import cv2
import numpy as np
from numba import njit


@njit
def image_convolution(matrix: np.ndarray, kernel: np.ndarray):
    # assuming kernel is symmetric and odd
    k_size = len(kernel)
    m_height, m_width, channels = matrix.shape

    # iterates through matrix, applies kernel, and sums
    output = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(2, m_height-2):
        for j in range(2, m_width-2):
            for ch in range(channels):
                output[i][j][ch] = np.sum(matrix[:, :, ch][i:k_size + i, j:k_size + j] * kernel, dtype=np.uint8)

    return output


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
    # todo: now h is range [0, 255]
    return hsv


def detect_logo():
    img = cv2.imread("./images/img2.jpeg")
    # img = img[100:300, 100:300]  # for development only
    # blur
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel = kernel / np.sum(kernel)
    img2 = image_convolution(img, kernel)

    # to hsv
    img2_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = bgr2hsv(img)



    # cv2.imshow("before", img)
    # cv2.imshow("result", img2)
    stacked = np.hstack((img2_cv, img2))
    cv2.imshow("result", stacked)


    if False:
        cv2.filter2D(img, -1, kernel, img)
        # v2.GaussianBlur(img, (3, 3), 0, img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # sharpening nie nie nie
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # cv2.filter2D(img, -1, kernel, img)

        # define range of red color in HSV
        lower_red = np.array([160, 50, 50])
        upper_red = np.array([180, 255, 255])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_red, upper_red)
        lower_red2 = np.array([0, 70, 50])
        upper_red2 = np.array([10, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        res2 = cv2.bitwise_and(img, img, mask=mask2)

        # closing
        # kernel = np.ones((5, 5), np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        res2 = cv2.dilate(res2, kernel)
        res2 = cv2.erode(res2, kernel)

        res = cv2.dilate(res, kernel)
        res = cv2.erode(res, kernel)

        # binary threshold
        for i_x, x in enumerate(res2):
            for i_y, y in enumerate(x):
                if not np.array_equal([0, 0, 0], y):
                    res2[i_x, i_y] = np.array([255, 255, 255])  # todo: binarize image

        for i_x, x in enumerate(res):
            for i_y, y in enumerate(x):
                if not np.array_equal([0, 0, 0], y):
                    res[i_x, i_y] = np.array([255, 255, 255])  # todo: binarize image
                # merge two images
                elif not np.array_equal([0, 0, 0], res2[i_x, i_y]):
                    res[i_x, i_y] = np.array([255, 255, 255])

        # stacking up all three images together
        stacked = np.hstack((res, res2))

        cv2.imshow('Result', cv2.resize(stacked, None, fx=1, fy=1))


if __name__ == '__main__':
    detect_logo()
    cv2.waitKey()
