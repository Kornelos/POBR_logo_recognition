import cv2
import numpy as np

from converters import bgr2hsv, hsv2bgr


# https://github.com/luktor99/pobr/blob/master/doc/sprawozdanie.pdf najlepsza inspiracja


def detect_logo():
    img = cv2.imread("./images/img3.jpeg")
    hsvImage = bgr2hsv(img)
    rows, cols, channels = hsvImage.shape
    for i in range(0, rows):
        for j in range(0, cols):
            pixel = hsvImage[i, j]
            # Red: H: 0 – 10 or 350 – 360 S: 0.3 – 1.0 V: 0.5 – 1.0;
            if (0 < pixel[0] < 10) or (350 < pixel[0] < 360) and (0.3 < pixel[1] < 1.0) and (0.5 < pixel[2] < 1.0):
                hsvImage[i, j] = pixel
            else:
                hsvImage[i, j] = [0, 0, 0]
    cv2.imshow('image', hsvImage)


def detect_logo_cv2():
    img = cv2.imread("./images/img5.jpeg")
    # cv2.medianBlur(img, 3, img)
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel = kernel / sum(kernel)
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
    detect_logo_cv2()
    cv2.waitKey()
