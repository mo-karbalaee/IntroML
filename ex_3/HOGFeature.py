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

    squared_gradient_x = gradient_x**2
    squared_gradient_y = gradient_y**2
    gradient_magnitude = np.sqrt(squared_gradient_x + squared_gradient_y)

    gradient_angle = np.arctan2(gradient_y, gradient_x)
    gradient_orientation = np.degrees(gradient_angle) % 180

    return gradient_magnitude, gradient_orientation


def buildCellHistograms(magnitude, orientation, cell_size=8, num_bins=9):
    if magnitude.shape != orientation.shape:
        raise ValueError("Magnitude and orientation must have the same shape.")

    image_height, image_width = magnitude.shape

    number_of_cells_y = image_height // cell_size
    number_of_cells_x = image_width // cell_size

    histogram_bin_width = 180.0 / num_bins

    cell_histograms = np.zeros(
        (number_of_cells_y, number_of_cells_x, num_bins),
        dtype=np.float32,
    )

    for cell_y in range(number_of_cells_y):
        for cell_x in range(number_of_cells_x):
            row_start = cell_y * cell_size
            row_end = row_start + cell_size

            column_start = cell_x * cell_size
            column_end = column_start + cell_size

            cell_magnitudes = magnitude[
                row_start:row_end,
                column_start:column_end,
            ]

            cell_orientations = orientation[
                row_start:row_end,
                column_start:column_end,
            ]

            floating_bin_indices = cell_orientations / histogram_bin_width

            lower_bin_indices = np.floor(floating_bin_indices).astype(int) % num_bins

            upper_bin_indices = (lower_bin_indices + 1) % num_bins

            upper_bin_weights = floating_bin_indices - np.floor(floating_bin_indices)

            lower_bin_weights = 1.0 - upper_bin_weights

            for histogram_bin in range(num_bins):
                lower_bin_mask = lower_bin_indices == histogram_bin

                upper_bin_mask = upper_bin_indices == histogram_bin

                lower_bin_contribution = (
                    cell_magnitudes * lower_bin_weights * lower_bin_mask
                )

                upper_bin_contribution = (
                    cell_magnitudes * upper_bin_weights * upper_bin_mask
                )

                lower_bin_sum = np.sum(lower_bin_contribution)

                upper_bin_sum = np.sum(upper_bin_contribution)

                cell_histograms[
                    cell_y,
                    cell_x,
                    histogram_bin,
                ] += (
                    lower_bin_sum + upper_bin_sum
                )

    return cell_histograms


def calculateHOG(
    img,
    cell_size=8,
    block_size=2,
    num_bins=9,
    eps=1e-6,
):
    """
    Compute a dense HOG descriptor with overlapping, normalized blocks.
    """
    gradient_magnitude, gradient_orientation = computeGradients(img)

    cell_histograms = buildCellHistograms(
        gradient_magnitude,
        gradient_orientation,
        cell_size,
        num_bins,
    )

    number_of_cells_y, number_of_cells_x, _ = cell_histograms.shape

    number_of_blocks_y = number_of_cells_y - block_size + 1

    number_of_blocks_x = number_of_cells_x - block_size + 1

    descriptor_blocks = []

    for block_y in range(number_of_blocks_y):
        for block_x in range(number_of_blocks_x):
            block_histograms = cell_histograms[
                block_y : block_y + block_size,
                block_x : block_x + block_size,
                :,
            ]

            flattened_block = block_histograms.flatten()

            squared_block = flattened_block**2
            squared_sum = np.sum(squared_block)

            first_norm = np.sqrt(squared_sum + eps**2)

            normalized_block = flattened_block / first_norm

            clipped_block = np.clip(
                normalized_block,
                0,
                0.2,
            )

            clipped_squared_block = clipped_block**2

            clipped_squared_sum = np.sum(clipped_squared_block)

            second_norm = np.sqrt(clipped_squared_sum + eps**2)

            final_block = clipped_block / second_norm

            descriptor_blocks.append(final_block)

    return np.concatenate(descriptor_blocks)
