"""
Created on 20.06.2025

@author: Linda Schneider
"""

import numpy as np
import cv2

# do not import more modules!
# Use OpenCV only for basic image operations such as resizing and thresholding.
# Use NumPy for the bounding box computation and for centering the symbol.
# Do not use contour detection or connected components here.


def simpleAlignment(img, size=128):
    """
    Align a grayscale symbol by centering its foreground on a fixed canvas.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    resized = cv2.resize(img, (size, size))

    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    foreground_pixels = np.argwhere(binary > 0)

    if foreground_pixels.size == 0:
        return np.zeros((size, size), dtype=np.uint8)

    r_min, c_min = foreground_pixels.min(axis=0)
    r_max, c_max = foreground_pixels.max(axis=0)

    cropped = resized[r_min : r_max + 1, c_min : c_max + 1]

    target = size // 2
    h, w = cropped.shape
    scale = target / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    symbol = cv2.resize(cropped, (new_w, new_h))

    canvas = np.zeros((size, size), dtype=np.uint8)
    r_start = (size - new_h) // 2
    c_start = (size - new_w) // 2
    canvas[r_start : r_start + new_h, c_start : c_start + new_w] = symbol

    return canvas
