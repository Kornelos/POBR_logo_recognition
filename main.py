from typing import Tuple

import cv2
import numpy as np

from converters import bgr2hsv
from detection import flood_fill
from processing import image_convolution, apply_threshold
from recognition import is_hm


def merge_boxes(p0, p1, p2, p3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # (j_min, i_min), (j_max, i_max)
    j_min = min(p0[0], p1[0], p2[0], p3[0])
    j_max = max(p0[0], p1[0], p2[0], p3[0])
    i_min = min(p0[1], p1[1], p2[1], p3[1])
    i_max = max(p0[1], p1[1], p2[1], p3[1])
    return (j_min, i_min), (j_max, i_max)


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

    # todo: closing (dilitate & erode) - maybe not necessary?
    kernel = np.ones((3, 3), np.uint8)
    # cv2.dilate(img2, kernel, img2)
    # cv2.erode(img2, kernel, img2)

    # flood fill & boxing
    detected = flood_fill(img2)
    merge_queue = []
    for idx, d in enumerate(detected):
        p1, p2 = d[2]  # (j_min, i_min), (j_max, i_max)
        hm, should_merge = is_hm(matrix=d[0], pixel_count=d[1])
        if hm and not should_merge:  # or True:
            cv2.rectangle(img, p1, p2, color=(0, 255, 0))
        if hm and should_merge:
            if merge_queue:
                # merge and print
                p1_prev, p2_prev = merge_queue.pop()
                p1, p2 = merge_boxes(p1_prev, p2_prev, p1, p2)
                cv2.rectangle(img, p1, p2, color=(0, 255, 0))
            else:
                merge_queue.append((p1, p2))
    cv2.imshow("result", np.hstack((img, img2)))


if __name__ == '__main__':
    logo_path = "./images/hm-logo.jpeg"

    detect_logo("./images/img1.jpeg")
    cv2.waitKey()
