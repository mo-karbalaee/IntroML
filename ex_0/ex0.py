import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:

    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
        image_path (str): Path to the input image.
        colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        self._colour_type: str = colour_type
        if self._colour_type == "BGR":
            self._image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)
        elif self._colour_type == "RGB":
            image_bgr: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self._image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            self._image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def get_image_data(self):
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        if self._colour_type == "Gray":
            plt.imshow(self._image, cmap="gray")
        else:
            plt.imshow(self._image)

        plt.show()
        pass

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
        image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)

        cv2.imwrite(total_image_path, self._image)
        pass

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        self._image = self._image[:, :, ::-1]

        self._colour_type = "RGB" if self._colour_type == "BGR" else "BGR"

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
        clip_min (int): Minimum image colour intensity.
        clip_max (int): Maximum image colour intensity.
        """
        self._image[self._image < clip_min] = clip_min
        self._image[self._image > clip_max] = clip_max

        pass

    def convert_to_grayscale(self, method: str = "lightness"):
        """
        Convert a colour image to a grayscale image.
        Write the different options from scratch.

        Args:
        method (str): Method for the colour conversion, either lightness, average or luminosity.
        """
        if method not in ["lightness", "average", "luminosity"]:
            raise ValueError("The given method is not supported!")
        if self._colour_type not in ["BGR", "RGB"]:
            raise ValueError("The function only works for colour images!")

        if self._colour_type == "RGB":
            r = self._image[:, :, 0]
            g = self._image[:, :, 1]
            b = self._image[:, :, 2]
        else:
            b = self._image[:, :, 0]
            g = self._image[:, :, 1]
            r = self._image[:, :, 2]

        if method == "lightness":
            self._image = (np.maximum(np.maximum(r, g), b) + np.minimum(np.minimum(r, g), b)) / 2

        if method == "average":
            self._image = (r + g + b) / 3

        if method == "luminosity":
            self._image = 0.21 * r + 0.73 * g + 0.07 * b

        self._colour_type = "Gray"

    def rotate_90_degrees(self):
        if self._image.ndim == 2:
            self._image = self._image[::-1, :].T
        else:
            self._image = self._image[::-1, :, :].transpose(1, 0, 2)

    def rotate_image(self, degrees: int = 0):
        """
        Rotate an image by a given angle (k * 90) clockwise.
        Do not use functions from external libraries apart from numpy.transpose.

        Args:
        degrees (int): Rotation angle.
        """
        if degrees % 90 != 0:
            raise ValueError("The provided rotation angle must be a multiple of 90!")

        k = degrees // 90
        k = k % 4

        if k == 0:
            pass
        elif k == 1:
            self.rotate_90_degrees()
        elif k == 2:
            self.rotate_90_degrees()
            self.rotate_90_degrees()
        else:
            self.rotate_90_degrees()
            self.rotate_90_degrees()
            self.rotate_90_degrees()

    def flip_vertically(self):
        if self._image.ndim == 2:
            self._image = self._image[::-1, :]
        else:
            self._image = self._image[::-1, :, :]

    def flip_horizontally(self):
        if self._image.ndim == 2:
            self._image = self._image[:, ::-1]
        else:
            self._image = self._image[:, ::-1, :]

    def flip_both(self):
        if self._image.ndim == 2:
            self._image = self._image[::-1, ::-1]
        else:
            self._image = self._image[::-1, ::-1, :]

    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
        flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        if flip_value == 0:
            self.flip_vertically()
        elif flip_value == 1:
            self.flip_horizontally()
        else:
            self.flip_both()

    def crop_center(self, new_height: int, new_width: int):
        """
        Crop the image to a given size around the center.
        Do not use functions from external libraries.

        Args:
        new_height (int): Height of the cropped image.
        new_width (int): Width of the cropped image.
        """
        center_h = self._image.shape[0] // 2
        center_w = self._image.shape[1] // 2
        h_plus = center_h + new_height // 2
        h_minus = center_h - new_height // 2
        w_plus = center_w + new_width // 2
        w_minus = center_w - new_width // 2

        if not isinstance(new_height, int) or not isinstance(new_width, int):
            raise TypeError("Height and width must be integers!")
        if new_height <= 0 or new_width <= 0:
            raise ValueError("Height and width must be positive!")
        if new_height > self._image.shape[0] or new_width > self._image.shape[1]:
            raise ValueError("Crop size cannot be larger than the original image!")

        self._image = self._image[h_minus: h_plus, w_minus: w_plus]

    def resize_image(self, new_height: int, new_width: int):
        """
        Resize an image to an arbitrary size using CV2.

        Args:
        new_height (int): Height of the resized image.
        new_width (int): Width of the resized image.
        """  # ToDo: Resize the image. Research the available options in CV2.


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="BGR")
