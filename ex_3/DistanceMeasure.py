'''
Distance measures for exercise 3.
'''

import numpy as np

# do not import more modules!
# Implement the formulas yourself using NumPy operations.
# Do not use external distance or metric libraries.

"""
This is used for pixel by pixel comparison of the images. 
"""
def mseDistance(imgA, imgB):
    """
    Compute the mean squared error between two equally sized grayscale images.
    """
    
    """
    The images should be of the same shape because the comparison is pixel by pixel. 
    """
    if imgA.shape != imgB.shape:
        raise ValueError("Images must have the same shape.")

    """
    We convert the images to floating point because the images are in uint8 now which means
    that when we subtract two images in the following lines, negative values can appear and
    the type unsigned int cannot store negative values so what happens here is that numpy will 
    wrap the value which mean to start the value from the other end when it overflows. 
    For example, 40 - 220 = -180 which cannot be stored so numpy does this: -180 + 256 = 76
    which is not what we want in MSE (mean squared error) formula. 
    """
    image_a = imgA.astype(np.float32)
    image_b = imgB.astype(np.float32)
    
    """
    Here is just the implementation of the MSE formula. It's actually so simple and the name
    suggests what it actually is. We take the differences and call it error because most often we are 
    comparing something against a ground truth and any deviation from the ground truth is considered 
    error. Then we raise the error by two which is the "squared" part in the name and finally we calculate
    the mean of these values. Pretty simple and pixel-wise comparison. 
    """
    difference = image_a - image_b
    squared_difference = difference**2
    mean_squared_error = np.mean(squared_difference)

    return mean_squared_error

"""
This is used for comparing the feature descriptor vector of two images. It's not that
you cannot perform pixel-wise comparison between two images using this function, but that
the naming of the arguments suggested that this is to be used for feature vectors not raw pixels. 
"""
def euclideanDistance(featureA, featureB):
    """
    Compute the Euclidean distance between two feature vectors.
    """
    if featureA.shape != featureB.shape:
        raise ValueError("Feature vectors must have the same shape.")
    
    
    """
    We are pretty much doing the same here but we don't average across the errors, but we sum them 
    and then take the square root of that sum. Nothing fancy or weird. Just look the formula of
    euclidean distance and you'll be good to go. 
    """
    difference = featureA - featureB
    squared_difference = difference**2
    sum_of_squares = np.sum(squared_difference)
    euclidean_distance = np.sqrt(sum_of_squares)

    return euclidean_distance
