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

"""
Converts the image to grayscale just in case.
We do this because for achieving a feature descriptor of the symbol, 
we only need the brightness values and dealing with three channels is not 
necessary for this task which is preprocessing for a classification task. 
"""
def _toGrayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


"""
We need to binarize the image because our goal is to distinguish the symbol 
inside each image and doing this is much easier with a binary image because 
the background will be removed and we only get the clean black pixels of the 
symbol itself. 
"""
def _binarize(img):
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

"""
The bounding box is the smallest rectangle that we can fit the symbol in. 
This methods calculates and returns the 4 vertices of that rectangle, but does not
crop the symbol to that bounding box!
"""
def _boundingBox(binary):
    """
    
    """
    foreground_pixels = np.argwhere(binary > 0)
    if foreground_pixels.size == 0:
        return None
    min_coords = foreground_pixels.min(axis=0)
    max_coords = foreground_pixels.max(axis=0)
    return min_coords[0], min_coords[1], max_coords[0], max_coords[1]

"""
Gets the bounding box coordinates and returns the cropped image resulting
into a clean box that contains the symbol. 
"""
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
    """
    We resize all the images to a fixed size because we need all of them to 
    cover up the exact same part of the canvas and since the images of the 
    dataset come in different resolutions, this would not be possible without resizing. 
    """
    resized = cv2.resize(grayscale, (size, size))
    binary = _binarize(resized)

    bbox = _boundingBox(binary)
    if bbox is None:
        """
        In case there is no symbol in the image, the bounding box function will return None,
        hence here we check for it and return a completely black image in that case. 
        """
        return np.zeros((size, size), dtype=np.uint8)

    cropped = _cropSymbol(resized, bbox)
    scaled = _scaleToTarget(cropped, size * 7 // 8)
    return _placeOnCanvas(scaled, size)
