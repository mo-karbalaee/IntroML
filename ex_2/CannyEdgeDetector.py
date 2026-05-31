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
    coords = np.arange(ksize) - center
    x, y = np.meshgrid(coords, coords)
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
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
    
    deg = np.degrees(theta) % 180
    angle = np.zeros_like(deg, dtype=int)
    angle[(deg >= 22.5) & (deg < 67.5)] = 45
    angle[(deg >= 67.5) & (deg < 112.5)] = 90
    angle[(deg >= 112.5) & (deg < 157.5)] = 135
 
    pad = np.pad(g, 1)
    n1 = np.where(angle == 0,  pad[1:-1, :-2],
         np.where(angle == 45, pad[2:,   :-2],
         np.where(angle == 90, pad[:-2,  1:-1],
                               pad[:-2,  :-2])))
    n2 = np.where(angle == 0,  pad[1:-1, 2:],
         np.where(angle == 45, pad[:-2,  2:],
         np.where(angle == 90, pad[2:,   1:-1],
                               pad[2:,   2:])))
 
    result = np.zeros_like(g)
    mask = (g >= n1) & (g >= n2)
    result[1:-1, 1:-1] = np.where(mask[1:-1, 1:-1], g[1:-1, 1:-1], 0)
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
        
    thresh = np.zeros_like(max_sup)
    thresh[max_sup > t_high] = 2
    thresh[(max_sup > t_low) & (max_sup <= t_high)] = 1
 
    strong = (thresh == 2)
    padded_strong = np.pad(strong.astype(np.uint8), 1)
    neighbor_has_strong = np.lib.stride_tricks.sliding_window_view(padded_strong, (3, 3)).any(axis=(-2, -1))
 
    result = np.zeros_like(max_sup)
    result[strong] = 255
    result[(thresh >= 1) & neighbor_has_strong] = 255
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