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


def _toGrayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _binarize(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _boundingBox(binary):
    foreground_pixels = np.argwhere(binary > 0)
    if foreground_pixels.size == 0:
        return None
    min_coords = foreground_pixels.min(axis=0)
    max_coords = foreground_pixels.max(axis=0)
    return min_coords[0], min_coords[1], max_coords[0], max_coords[1]


def _cropSymbol(img, bbox):
    min_row, min_col, max_row, max_col = bbox
    return img[min_row : max_row + 1, min_col : max_col + 1]


def _scaleToTarget(symbol, target_size):
    h, w = symbol.shape
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(symbol, (new_w, new_h))


def _placeOnCanvas(symbol, size):
    canvas = np.zeros((size, size), dtype=np.uint8)
    h, w = symbol.shape
    row_start = (size - h) // 2
    col_start = (size - w) // 2
    canvas[row_start : row_start + h, col_start : col_start + w] = symbol
    return canvas


def simpleAlignment(img, size=128):
    """
    Align a grayscale symbol by centering its foreground on a fixed canvas.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    grayscale = _toGrayscale(img)
    resized = cv2.resize(grayscale, (size, size))
    binary = _binarize(resized)

    bbox = _boundingBox(binary)
    if bbox is None:
        return np.zeros((size, size), dtype=np.uint8)

    cropped = _cropSymbol(resized, bbox)
    scaled = _scaleToTarget(cropped, size * 7 // 8)
    return _placeOnCanvas(scaled, size)
