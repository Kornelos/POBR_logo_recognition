import cv2
import numpy as np

from converters import bgr2hsv
from detection import flood_fill
from processing import image_convolution, apply_threshold

PIXEL_COUNT_MIN = 1000


def detect_logo(path):
    img = cv2.imread(path)
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
    # img2_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img2 = bgr2hsv(img2)

    # print(_bgr2hsv_pixel_fast(np.array([201, 38, 53])))
    # threshold
    img2 = apply_threshold(img2, np.uint8(171))

    # todo: closing (dilitate & erode) not necessary
    # kernel = np.ones((3, 3), np.uint8)
    # cv2.dilate(img2,kernel,img2)

    # flood fill & boxing
    detected = flood_fill(img2)

    for d in detected:
        p1, p2 = d[2]  # (j_min, i_min), (j_max, i_max)
        cv2.rectangle(img, p1, p2, color=(0, 255, 0))
    cv2.imshow("result", img)


if __name__ == '__main__':
    detect_logo("./images/img5.jpeg")
    cv2.waitKey()
