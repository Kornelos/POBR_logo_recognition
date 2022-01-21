from typing import Tuple

import numpy as np
from cv2 import imshow, rectangle, waitKey, imread

from converters import bgr2hsv
from detection import flood_fill
from processing import image_convolution, apply_threshold, dilate, erode
from recognition import is_hm

SHOW_THRESHOLD = False


def merge_boxes(p0, p1, p2, p3) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    j_min = min(p0[0], p1[0], p2[0], p3[0])
    j_max = max(p0[0], p1[0], p2[0], p3[0])
    i_min = min(p0[1], p1[1], p2[1], p3[1])
    i_max = max(p0[1], p1[1], p2[1], p3[1])
    return (j_min, i_min), (j_max, i_max)


def detect_logo(path, should_close):
    img = imread(path)

    # blur
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    kernel = kernel / np.sum(kernel)
    img2 = image_convolution(img, kernel)

    # to hsv
    img2 = bgr2hsv(img2)

    # threshold
    img2 = apply_threshold(img2)

    # closing (dilitate & erode) 3x3 kernel
    if should_close:
        img2 = dilate(img2)
        img2 = erode(img2)

    # flood fill & boxing
    detected = flood_fill(img2)
    merge_queue = []
    p1, p2 = (0, 0)
    for idx, d in enumerate(detected):
        p1, p2 = d[2]
        hm, should_merge = is_hm(matrix=d[0], pixel_count=d[1])
        if hm and not should_merge:
            rectangle(img, p1, p2, color=(0, 255, 0), thickness=3)
        if hm and should_merge:
            if merge_queue:
                # merge and print
                p1_prev, p2_prev = merge_queue.pop()

                p1, p2 = merge_boxes(p1_prev, p2_prev, p1, p2)
                rectangle(img, p1, p2, color=(0, 255, 0), thickness=3)
            else:
                merge_queue.append((p1, p2))
    if merge_queue:
        rectangle(img, p1, p2, color=(0, 255, 0), thickness=3)

    if SHOW_THRESHOLD:
        imshow(path, np.hstack((img, img2)))
    else:
        imshow(path, img)


if __name__ == '__main__':
    logo_path = "./images/hm-logo.jpeg"
    detect_logo("./images/img1.jpeg", False)
    detect_logo("./images/img2.jpeg", False)
    detect_logo("./images/img3.jpeg", True)
    waitKey()
