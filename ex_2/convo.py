from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    for x in range(ksize):
        for y in range(ksize):
            dx = x - center
            dy = y - center
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
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
    for i in range(I):
        for j in range(J):
            val = 0.0
            for u in range(U):
                for v in range(V):
                    val += k[u, v] * padded[i + (U - 1 - u), j + (V - 1 - v)]
            result[i, j] = val
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