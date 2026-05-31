import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def extract_region(padded_image: np.ndarray, center_row: int, center_col: int, window_size: int) -> np.ndarray:
    # The function receives a padded image (pad_image) and the current pixel of our padded image.

    half = window_size // 2
    return padded_image[center_row - half: center_row + half + 1,
                        center_col - half: center_col + half + 1]


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    # Pad the image with zeros.

    return np.pad(image, pad_width=padding_size, mode='constant', constant_values=0)


def erode_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply erosion on the given image using the structuring element.

    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."
 
    padded = pad_image(image, se_size // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (se_size, se_size))
    active = structuring_element == 1
    return np.where(np.all(windows[:, :, active] == 1, axis=-1), 1, 0).astype(np.uint8)


def dilate_binary(image: np.ndarray, structuring_element: np.ndarray) -> np.ndarray:
    # Apply dilation on the given image using the structuring element.

    se_size = structuring_element.shape[0]
    assert se_size == structuring_element.shape[1], "SE must be quadratic."
    assert se_size % 2 == 1, "SE size must be uneven."
 
    padded = pad_image(image, se_size // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (se_size, se_size))
    active = structuring_element == 1
    return np.where(np.any(windows[:, :, active] == 1, axis=-1), 1, 0).astype(np.uint8)


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
    # Load the image and binarize it again with a simple threshold.

    img = Image.open(filepath).convert('L')
    arr = np.array(img, dtype=np.uint8)
    binary_arr = (arr > 128).astype(np.uint8)
    return binary_arr


def save_binary(image_array: np.ndarray, filepath: str):
    # Save the binary image.

    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(filepath)


def show_image(image_array: np.ndarray, title: str = ""):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Paths.
    raw_erosion_image_path = 'data/erosion_image_raw.png'
    raw_dilation_image_path = 'data/dilation_image_raw.png'
    erosion_out_path = 'data/erosion_output.png'
    dilation_out_path = 'data/dilation_output.png'

    # Load images.
    erosion_input = load_binary(raw_erosion_image_path)
    dilation_input = load_binary(raw_dilation_image_path)

    # Structuring element.
    SE = np.ones((5, 5), dtype=np.uint8)

    # Erosion.
    eroded = erode_binary(erosion_input, SE)
    save_binary(eroded, erosion_out_path)
    show_image(eroded, "Erosion Output")

    # Dilation.
    dilated = dilate_binary(dilation_input, SE)
    save_binary(dilated, dilation_out_path)
    show_image(dilated, "Dilation Output")