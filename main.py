import cv2
import numpy as np


def low_pass(img):
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel = kernel / sum(kernel)

    for i_x, x in enumerate(img):
        for i_y, y in enumerate(x):
            if not np.array_equal([0, 0, 0], y):
                img[i_x, i_y] = np.array([255, 255, 255])  # todo: binarize image


def image_convolution(matrix, kernel):
    # assuming kernel is symmetric and odd
    k_size = len(kernel)
    m_height, m_width, channels = matrix.shape
    # padded = []
    # for i in range(channels):
    #     padded.append(np.pad(matrix[:, :, i], (k_size - 1, k_size - 1)))

    # iterates through matrix, applies kernel, and sums
    output = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(2, m_height-2):
        for j in range(2, m_width-2):
            for ch in range(channels):
                output[i][j][ch] = np.sum(matrix[:, :, ch][i:k_size + i, j:k_size + j] * kernel, dtype=np.uint8)

    # output = np.array(output).reshape((m_height, m_width))
    return output


def detect_logo():
    img = cv2.imread("./images/img2.jpeg")
    img = img[100:300, 100:300]  # for development only

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel = kernel / np.sum(kernel)
    img2 = image_convolution(img, kernel)
    cv2.imshow("before", img)
    cv2.imshow("result", img2)
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
