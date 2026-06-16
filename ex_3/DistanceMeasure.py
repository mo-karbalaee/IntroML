'''
Distance measures for exercise 3.
'''

import numpy as np

# do not import more modules!
# Implement the formulas yourself using NumPy operations.
# Do not use external distance or metric libraries.

def mseDistance(imgA, imgB):
    """
    Compute the mean squared error between two equally sized grayscale images.
    """
    
    if imgA.shape != imgB.shape:
        raise ValueError("Images must have the same shape.")

    image_a = imgA.astype(np.float32)
    image_b = imgB.astype(np.float32)

    difference = image_a - image_b
    squared_difference = difference**2
    mean_squared_error = np.mean(squared_difference)

    return mean_squared_error

def euclideanDistance(featureA, featureB):
    """
    Compute the Euclidean distance between two feature vectors.
    """
    if featureA.shape != featureB.shape:
        raise ValueError("Feature vectors must have the same shape.")
    
    difference = featureA - featureB
    squared_difference = difference**2
    sum_of_squares = np.sum(squared_difference)
    euclidean_distance = np.sqrt(sum_of_squares)

    return euclidean_distance
