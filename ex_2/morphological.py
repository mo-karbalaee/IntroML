import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    half = window_size // 2
    return padded_image[center_row - half: center_row + half + 1,
                        center_col - half: center_col + half + 1]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    radius = se_size // 2
    padded = pad_image(image, radius)
    output = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            region = extract_region(padded, i + radius, j + radius, se_size)
            if np.all(region[structuring_element == 1] == 1):
                output[i, j] = 1

    return output


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."

    radius = se_size // 2
    padded = pad_image(image, radius)
    output = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            region = extract_region(padded, i + radius, j + radius, se_size)
            if np.any(region[structuring_element == 1] == 1):
                output[i, j] = 1

    return output


def open_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = input_image.copy()
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
    return result


def close_binary(input_image: np.ndarray, structuring_element: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = input_image.copy()
    for _ in range(iterations):
        result = dilate_binary(result, structuring_element)
    for _ in range(iterations):
        result = erode_binary(result, structuring_element)
    return result


def load_binary(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    raw_erosion_image_path = 'data/erosion_image_raw.png'
    raw_dilation_image_path = 'data/dilation_image_raw.png'
    erosion_out_path = 'data/erosion_output.png'
    dilation_out_path = 'data/dilation_output.png'

    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    SE = np.ones((5, 5), dtype=np.uint8)

    eroded = open_binary(erosion_input, SE, iterations=3)
    save_binary(eroded, erosion_out_path)
    show_image(eroded, "Erosion Output")

    dilated = close_binary(dilation_input, SE, iterations=3)
    save_binary(dilated, dilation_out_path)
    show_image(dilated, "Dilation Output")