"""
Histogram of Oriented Gradients utilities for exercise 3.
"""

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

    img_float = img.astype(np.float32)
    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.degrees(np.arctan2(np.abs(gy), gx)) % 180

    return magnitude, orientation


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    h, w = magnitude.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    bin_width = 180.0 / num_bins

    histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag = magnitude[
                cy * cell_size : (cy + 1) * cell_size,
                cx * cell_size : (cx + 1) * cell_size,
            ]
            cell_ori = orientation[
                cy * cell_size : (cy + 1) * cell_size,
                cx * cell_size : (cx + 1) * cell_size,
            ]
            bin_indices = (cell_ori / bin_width).astype(int) % num_bins
            for b in range(num_bins):
                histograms[cy, cx, b] = cell_mag[bin_indices == b].sum()

    return histograms


def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    magnitude, orientation = computeGradients(img)
    histograms = buildCellHistograms(magnitude, orientation, cell_size, num_bins)

    n_cells_y, n_cells_x, _ = histograms.shape
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    descriptor = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = histograms[by:by+block_size, bx:bx+block_size, :].flatten()
            norm = np.sqrt(np.sum(block**2) + eps**2)
            descriptor.append(block / norm)

    return np.concatenate(descriptor)
