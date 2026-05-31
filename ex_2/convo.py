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
    pad_u = U // 2
    pad_v = V // 2
    padded = np.zeros((I + 2 * pad_u, J + 2 * pad_v))
    padded[pad_u:pad_u + I, pad_v:pad_v + J] = arr
    result = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            val = 0.0
            for u in range(-(U // 2), U - U // 2):
                for v in range(-(V // 2), V - V // 2):
                    ku = u + U // 2
                    kv = v + V // 2
                    val += k[ku, kv] * padded[i - u + pad_u, j - v + pad_v]
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