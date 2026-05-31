import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

#
# NO MORE MODULES ALLOWED
#

def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """

    center = ksize // 2
    kernel = np.zeros((ksize, ksize))
    for x in range(ksize):
        for y in range(ksize):
            dx = x - center
            dy = y - center
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    filtered = convolve(img_in, kernel).astype(int)
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
        
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]])
    gx = convolve(img_in, Gx).astype(int)
    gy = convolve(img_in, Gy).astype(int)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
        
    g = np.sqrt(gx**2 + gy**2).astype(int)
    theta = np.arctan2(gy, gx)
    return g, theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
        
    deg = np.degrees(angle) % 180
    if deg < 22.5 or deg >= 157.5:
        return 0
    elif deg < 67.5:
        return 45
    elif deg < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    
    rows, cols = g.shape
    result = np.zeros_like(g)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = convertAngle(theta[i, j])
            if angle == 0:
                n1, n2 = g[i, j-1], g[i, j+1]
            elif angle == 45:
                n1, n2 = g[i+1, j-1], g[i-1, j+1]
            elif angle == 90:
                n1, n2 = g[i-1, j], g[i+1, j]
            else:
                n1, n2 = g[i-1, j-1], g[i+1, j+1]
            if g[i, j] >= n1 and g[i, j] >= n2:
                result[i, j] = g[i, j]
    return result


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
        
    rows, cols = max_sup.shape
    thresh = np.zeros_like(max_sup)
    thresh[max_sup > t_high] = 2
    thresh[(max_sup > t_low) & (max_sup <= t_high)] = 1
    result = np.zeros_like(max_sup)
    for i in range(rows):
        for j in range(cols):
            if thresh[i, j] == 2:
                result[i, j] = 255
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if thresh[ni, nj] >= 1:
                                result[ni, nj] = 255
    return result



def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result