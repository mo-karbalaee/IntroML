from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    center = ksize // 2
    coords = np.arange(ksize) - center
    x, y = np.meshgrid(coords, coords)
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def slow_convolve(arr, k):
    U, V = k.shape
    I, J = arr.shape
    pad_top = U // 2
    pad_bot = U - 1 - pad_top
    pad_left = V // 2
    pad_right = V - 1 - pad_left
    padded = np.zeros((I + pad_top + pad_bot, J + pad_left + pad_right))
    padded[pad_top:pad_top + I, pad_left:pad_left + J] = arr
    result = np.zeros((I, J))
    for u in range(U):
        for v in range(V):
            result += k[U - 1 - u, V - 1 - v] * padded[u:u+I, v:v+J]
    return result


if __name__ == '__main__':
    k = make_kernel(9, 2)

    im = np.array(Image.open('./data/input1.jpg'))

    if im.ndim == 3:
        blurred = np.stack([slow_convolve(im[:, :, c], k) for c in range(im.shape[2])], axis=2)
    else:
        blurred = slow_convolve(im, k)

    unsharp_mask = im - blurred
    sharpened = im + unsharp_mask
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    Image.fromarray(sharpened).save('output.jpg')