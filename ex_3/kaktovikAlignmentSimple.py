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
    """
    Converts the image to grayscale just in case.
    We do this because for achieving a feature descriptor of the symbol, 
    we only need the brightness values and dealing with three channels is not 
    necessary for this task which is preprocessing for a classification task. 
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _binarize(img):
    """
    We need to binarize the image because our goal is to distinguish the symbol 
    inside each image and doing this is much easier with a binary image because 
    the background will be removed and we only get the clean black pixels of the 
    symbol itself. 
    """
    """
    # Args
    1. The image
    2. The threshold which gets ignored here because we are sending the otsu flags
    3. The maximum pixel value. 
    4. The flags. Two flags have been passed. THRESH_OTSU specifies the thresholding 
        algorithm for binarization. THRESH_BINARY_INV flag is about how we decide 
        pixel values. This flag says, anything below the threshold should become white. 
        That's the inverse part. The binary part specifies that the decision is a simple if 
        statement because there are more complex ways of doing this as well. We make the symbol
        pixels white because many techniques actually care for non-zero values not zero values
        so dealing with zero values is not a common practice in image processing so we invert the 
        image during the binarization. 
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    """
    OpenCV's thresholding function returns two values, the threshold value that it 
    computed during the Otsu thresholding, remember ex1?, and the binarized image itself.
    Since we don't need the threshold value here, so after unpacking the function we 
    threw away the threshold itself and only stored the binarized image. 
    """
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
