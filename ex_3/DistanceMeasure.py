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

    diff = imgA.astype(np.float32) - imgB.astype(np.float32)
    return np.mean(diff**2)


def euclideanDistance(featureA, featureB):
    """
    Compute the Euclidean distance between two feature vectors.
    """
    if featureA.shape != featureB.shape:
        raise ValueError("Feature vectors must have the same shape.")

    diff = featureA - featureB
    return np.sqrt(np.sum(diff**2))
