'''
Histogram of Oriented Gradients utilities for exercise 3.
'''

import numpy as np
import cv2

# do not import more modules!
# You may use cv2.Sobel for the derivatives.
# Compute magnitudes, orientations, histogram binning, and block normalization yourself with NumPy.
# Do not use cv2.HOGDescriptor or any other ready-made HOG implementation.


def computeGradients(img):
    """
    Compute gradient magnitudes and unsigned orientations in degrees.
    """
    if img is None:
        raise ValueError("Input image must not be None.")

    # TODO: compute Sobel derivatives, magnitudes, and orientations.
    # Allowed: cv2.Sobel for the x/y derivatives and NumPy for the remaining computations.
    # Not allowed: any ready-made HOG or feature extraction implementation.
    pass


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    # TODO: divide the image into cells and accumulate magnitudes into bins.
    # Use NumPy indexing/loops to implement the histogram accumulation yourself.
    # Do not call a library routine that directly computes cell histograms for HOG.
    pass


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    # TODO: compute the final descriptor from your own cell histograms.
    # Implement the block normalization and concatenation yourself with NumPy.
    # Do not use cv2.HOGDescriptor, skimage.feature.hog, or similar helpers.
    pass
