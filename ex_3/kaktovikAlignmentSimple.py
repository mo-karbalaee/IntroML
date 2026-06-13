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

    if len(img.shape) == 3:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = img

    resized_image = cv2.resize(grayscale_image, (size, size))

    _, binary_image = cv2.threshold(
        resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground_pixels = np.argwhere(binary_image > 0)

    if foreground_pixels.size == 0:
        empty_canvas = np.zeros((size, size), dtype=np.uint8)
        return empty_canvas

    minimum_coordinates = foreground_pixels.min(axis=0)
    maximum_coordinates = foreground_pixels.max(axis=0)

    minimum_row = minimum_coordinates[0]
    minimum_column = minimum_coordinates[1]
    maximum_row = maximum_coordinates[0]
    maximum_column = maximum_coordinates[1]

    cropped_symbol = resized_image[
        minimum_row : maximum_row + 1, minimum_column : maximum_column + 1
    ]

    target_size = size * 7 // 8

    symbol_height, symbol_width = cropped_symbol.shape
    largest_dimension = max(symbol_height, symbol_width)

    scale_factor = target_size / largest_dimension

    new_height = int(symbol_height * scale_factor)
    new_width = int(symbol_width * scale_factor)

    resized_symbol = cv2.resize(cropped_symbol, (new_width, new_height))

    aligned_image = np.zeros((size, size), dtype=np.uint8)

    row_start = (size - new_height) // 2
    column_start = (size - new_width) // 2

    row_end = row_start + new_height
    column_end = column_start + new_width

    aligned_image[row_start:row_end, column_start:column_end] = resized_symbol

    return aligned_image
