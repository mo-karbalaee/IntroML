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

    image_float = img.astype(np.float32)

    gradient_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=1)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180

    return gradient_magnitude, gradient_orientation


def _interpolateBins(cell_magnitudes, cell_orientations, num_bins):
    histogram_bin_width = 180.0 / num_bins

    floating_bin_indices = cell_orientations / histogram_bin_width
    lower_bin_indices = np.floor(floating_bin_indices).astype(int) % num_bins
    upper_bin_indices = (lower_bin_indices + 1) % num_bins
    upper_bin_weights = floating_bin_indices - np.floor(floating_bin_indices)
    lower_bin_weights = 1.0 - upper_bin_weights

    return lower_bin_indices, upper_bin_indices, lower_bin_weights, upper_bin_weights


def _computeCellHistogram(cell_magnitudes, cell_orientations, num_bins):
    lower_indices, upper_indices, lower_weights, upper_weights = _interpolateBins(
        cell_magnitudes, cell_orientations, num_bins
    )

    histogram = np.zeros(num_bins, dtype=np.float32)

    for b in range(num_bins):
        histogram[b] += np.sum(cell_magnitudes * lower_weights * (lower_indices == b))
        histogram[b] += np.sum(cell_magnitudes * upper_weights * (upper_indices == b))

    return histogram


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    """
    Accumulate orientation histograms for each cell.
    """
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    image_height, image_width = magnitude.shape
    number_of_cells_y = image_height // cell_size
    number_of_cells_x = image_width // cell_size

    cell_histograms = np.zeros(
        (number_of_cells_y, number_of_cells_x, num_bins),
        dtype=np.float32,
    )

    for cell_y in range(number_of_cells_y):
        for cell_x in range(number_of_cells_x):
            row_start = cell_y * cell_size
            col_start = cell_x * cell_size

            cell_magnitudes = magnitude[
                row_start : row_start + cell_size, col_start : col_start + cell_size
            ]
            cell_orientations = orientation[
                row_start : row_start + cell_size, col_start : col_start + cell_size
            ]

            cell_histograms[cell_y, cell_x] = _computeCellHistogram(
                cell_magnitudes, cell_orientations, num_bins
            )

    return cell_histograms


def _normalizeBlock(block_histograms, eps):
    flat = block_histograms.flatten()

    norm1 = np.sqrt(np.sum(flat**2) + eps**2)
    normalized = flat / norm1

    clipped = np.clip(normalized, 0, 0.2)

    norm2 = np.sqrt(np.sum(clipped**2) + eps**2)
    return clipped / norm2


def _extractBlocks(cell_histograms, block_size):
    number_of_cells_y, number_of_cells_x, _ = cell_histograms.shape
    number_of_blocks_y = number_of_cells_y - block_size + 1
    number_of_blocks_x = number_of_cells_x - block_size + 1

    for block_y in range(number_of_blocks_y):
        for block_x in range(number_of_blocks_x):
            yield cell_histograms[
                block_y : block_y + block_size,
                block_x : block_x + block_size,
                :,
            ]

"""
Compute a dense HOG descriptor with overlapping, normalized blocks.
Dense refers to the fact that HOG is calculated for all the cells because there are
some methods that only work on specific cells. An overlapping normalized block is like
a kernel that slides across the pixels of a cell and normalizes the values under that block. 
As a reminder, normalization is the process of making a bunch of values add up to one. In other words, scaling
some values to fit within a specific range. It is called overlapping because during the sliding, 
blocks have shared pixels, hence they overlap. This overlapping feature is good for the robustness of the final
descriptor because each block gets normalized multiple times and this makes the descriptor more robust compared
to just performing the normalization once per cell. 

# Args
1. The raw image. 
2. The size of the cells. For example, the image will be divided into cells of 8 * 8 size. Not the number of cells!
3. The size of the square sliding block within each cell. 
4. Number of histogram bins of the orientations. 
5. The eps is a small value to prevent division by zero in the normalization process. 

For example, an image of size 128, will get 128 / 8 = 16 cells per row hence 16 * 16 cells in total. 
"""
def calculateHOG(img, cell_size=8, block_size=2, num_bins=9, eps=1e-6):
    """
    First we calculate the magnitude and orientation gradients which are the base values for doing HOG.
    The name speaks for itself. Histogram of Gradients. In order to make a histogram out of gradients with 
    a special algorithm, we first need to have the gradients. That's what's happening here. As a quick reminder, 
    gradients are nothing but measurements for how much brightness changes. The idea is that edges are nothing, but
    relatively fast brightness changes across an image. 
    """
    gradient_magnitude, gradient_orientation = computeGradients(img)


    """
    For each cell we calculate the histogram. 
    """
    cell_histograms = buildCellHistograms(
        gradient_magnitude, gradient_orientation, cell_size, num_bins
    )
    
    
    """
    Within each cell we extract the blocks and apply normalization to it. This gives us a list of 
    descriptor blocks. 
    """
    descriptor_blocks = [
        _normalizeBlock(block, eps)
        for block in _extractBlocks(cell_histograms, block_size)
    ]

    """
    The final feature descriptor is nothing but the concatenation of each cell's feature descriptor.
    Concatenation is the process of putting lists back to back. Nothing fancy here. 
    """
    return np.concatenate(descriptor_blocks)
